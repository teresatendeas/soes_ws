import enum
import math
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Int32
from soes_msgs.msg import JointTargets, CupcakeCenters


class Phase(enum.Enum):
    WAIT        = 0
    MOVE_C1     = 1
    SWIRL_C1    = 2
    MOVE_C2     = 3
    SWIRL_C2    = 4
    MOVE_C3     = 5
    SWIRL_C3    = 6


class RoboHandNode(Node):
    """
    Resolved-rate IK (differential kinematics) controller for a 4-DOF arm:
    q = [yaw, pitch1, pitch2, pitch3]

    - Planar link stack after a base yaw.
    - FK is geometric (simple chain), Jacobian is numerical (finite differences).
    - Drives to a Cartesian target using: qdot = J^T (J J^T + λ^2 I)^(-1) * v
      where v = Kp * (x_des - x).
    """
    def __init__(self):
        super().__init__('soes_robothand')

        # ---------- Parameters ----------
        self.declare_parameter('rate_hz', 50.0)
        self.declare_parameter('pos_tol_m', 0.003)          # 3 mm tolerance
        self.declare_parameter('settle_s', 0.20)            # hold within tol before switching phase
        self.declare_parameter('lift_height_m', 0.03)       # +3 cm for "SWIRL" step

        # link lengths (meters) — tune to your arm
        self.declare_parameter('link_lengths_m', [0.12, 0.12, 0.10])  # [L1, L2, L3]

        # IK gains & limits
        self.declare_parameter('kp_cart', 2.0)              # proportional gain in task space [1/s]
        self.declare_parameter('damping_lambda', 0.01)      # damped least squares
        self.declare_parameter('qdot_limit_rad_s', [1.5, 1.5, 1.5, 1.5])
        self.declare_parameter('q_min_rad', [-math.pi, -math.pi/2, -math.pi/2, -math.pi/2])
        self.declare_parameter('q_max_rad', [ math.pi,  math.pi/2,  math.pi/2,  math.pi/2])

        self.rate_hz      = float(self.get_parameter('rate_hz').value)
        self.dt           = 1.0 / self.rate_hz
        self.pos_tol      = float(self.get_parameter('pos_tol_m').value)
        self.settle_s     = float(self.get_parameter('settle_s').value)
        self.lift_h       = float(self.get_parameter('lift_height_m').value)
        self.links        = np.array(self.get_parameter('link_lengths_m').value, dtype=float)
        self.kp           = float(self.get_parameter('kp_cart').value)
        self.lmbda        = float(self.get_parameter('damping_lambda').value)
        self.qdot_lim     = np.array(self.get_parameter('qdot_limit_rad_s').value, dtype=float)
        self.q_min        = np.array(self.get_parameter('q_min_rad').value, dtype=float)
        self.q_max        = np.array(self.get_parameter('q_max_rad').value, dtype=float)

        # ---------- IO ----------
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.index_sub = self.create_subscription(Int32, '/state/active_index', self._on_index, 10)
        self.center_sub = self.create_subscription(CupcakeCenters, '/vision/centers', self._on_centers, qos)
        self.targets_pub = self.create_publisher(JointTargets, '/arm/joint_targets', 10)

        # ---------- Runtime ----------
        # joint estimate (open-loop here; replace with feedback when encoders are available)
        self.q = np.zeros(4, dtype=float)
        self.active_index = -1
        self.centers: Optional[list[Tuple[float,float,float]]] = None
        self.phase = Phase.WAIT
        self.phase_start = self.get_clock().now()
        self.last_within_tol = None  # time when we first got within tol (for settle)
        self.des_xyz = None

        # timer
        self.timer = self.create_timer(self.dt, self._tick)
        self.get_logger().info('soes_robothand (resolved-rate IK) started.')

    # ================= Subs =================
    def _on_index(self, msg: Int32):
        prev = self.active_index
        self.active_index = int(msg.data)
        self.get_logger().info(f"active_index: {prev} -> {self.active_index}")
        self._align_phase_with_index()

    def _on_centers(self, msg: CupcakeCenters):
        self.centers = [(p.x, p.y, p.z) for p in msg.centers]
        if len(self.centers) < 3:
            self.get_logger().warn("centers < 3; waiting for full three targets")

    # ================= Helpers =================
    def _align_phase_with_index(self):
        # When the state node selects 0/1/2, start that center's MOVE. -1 -> WAIT.
        if self.active_index == 0:
            self._enter(Phase.MOVE_C1, self._target_xyz(0, lifted=False))
        elif self.active_index == 1:
            self._enter(Phase.MOVE_C2, self._target_xyz(1, lifted=False))
        elif self.active_index == 2:
            self._enter(Phase.MOVE_C3, self._target_xyz(2, lifted=False))
        else:
            self._enter(Phase.WAIT, None)

    def _enter(self, new_phase: Phase, xyz: Optional[Tuple[float,float,float]]):
        self.phase = new_phase
        self.phase_start = self.get_clock().now()
        self.last_within_tol = None
        self.des_xyz = np.array(xyz, dtype=float) if xyz is not None else None
        self.get_logger().info(f"[ROBOHAND] -> {self.phase.name}  des={self.des_xyz if xyz is not None else None}")

    def _elapsed(self) -> float:
        return (self.get_clock().now() - self.phase_start).nanoseconds * 1e-9

    def _target_xyz(self, idx: int, lifted: bool) -> Optional[Tuple[float,float,float]]:
        if self.centers is None or idx not in (0,1,2):
            return None
        x, y, z = self.centers[idx]
        if lifted:
            z += self.lift_h
        return (x, y, z)

    # ================= Kinematics =================
    def fk_xyz(self, q: np.ndarray) -> np.ndarray:
        """
        Very simple FK model:
          q = [yaw, th1, th2, th3]
          Base yaw rotates the planar chain in XY.
          Planar chain (th1, th2, th3) extends in a plane whose normal is base yaw.
          We compute planar X'Z (with Y'=0), then rotate by yaw around Z to world XY.
        """
        yaw, th1, th2, th3 = q
        L1, L2, L3 = self.links

        # planar forward (x', z)
        x_p = L1*math.cos(th1) + L2*math.cos(th1+th2) + L3*math.cos(th1+th2+th3)
        z   = L1*math.sin(th1) + L2*math.sin(th1+th2) + L3*math.sin(th1+th2+th3)
        # rotate by yaw about world Z to (x,y)
        cosy, siny = math.cos(yaw), math.sin(yaw)
        x = cosy * x_p
        y = siny * x_p
        return np.array([x, y, z], dtype=float)

    def jacobian_num(self, q: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """
        Numerical 3x4 Jacobian via finite differences.
        """
        f0 = self.fk_xyz(q)
        J = np.zeros((3, 4), dtype=float)
        for i in range(4):
            dq = np.zeros_like(q)
            dq[i] = eps
            fi = self.fk_xyz(q + dq)
            J[:, i] = (fi - f0) / eps
        return J

    # ================= IK step =================
    def ik_step(self, des_xyz: np.ndarray) -> None:
        cur_xyz = self.fk_xyz(self.q)
        err = des_xyz - cur_xyz
        err_norm = float(np.linalg.norm(err))

        # settle logic
        if err_norm <= self.pos_tol:
            if self.last_within_tol is None:
                self.last_within_tol = self.get_clock().now()
        else:
            self.last_within_tol = None

        # Task-space velocity
        v = self.kp * err  # [m/s] toward target

        # DLS pseudo-inverse: qdot = J^T (J J^T + λ^2 I)^-1 v
        J = self.jacobian_num(self.q)
        JJt = J @ J.T
        I3 = np.eye(3)
        qdot = J.T @ np.linalg.solve(JJt + (self.lmbda**2) * I3, v)

        # limit joint speeds
        qdot = np.clip(qdot, -self.qdot_lim, self.qdot_lim)

        # integrate & clamp to limits
        self.q = self.q + qdot * self.dt
        self.q = np.clip(self.q, self.q_min, self.q_max)

        # publish desired joint positions (position mode)
        out = JointTargets()
        out.position = [float(a) for a in self.q]
        out.velocity = [0.0, 0.0, 0.0, 0.0]
        out.use_velocity = False
        self.targets_pub.publish(out)

    # ================= Main loop =================
    def _tick(self):
        if self.centers is None:
            return

        # Run IK toward current phase target (if any)
        if self.des_xyz is not None:
            self.ik_step(self.des_xyz)

        # Phase transitions (index-driven + settle-on-target)
        within_settle = (
            self.last_within_tol is not None and
            (self.get_clock().now() - self.last_within_tol) >= Duration(seconds=self.settle_s)
        )

        if self.phase == Phase.MOVE_C1 and within_settle:
            self._enter(Phase.SWIRL_C1, self._target_xyz(0, lifted=True))

        elif self.phase == Phase.SWIRL_C1 and within_settle:
            # Wait for /state/active_index to advance to 1 (external state node)
            self._enter(Phase.WAIT, None)

        elif self.phase == Phase.MOVE_C2 and within_settle:
            self._enter(Phase.SWIRL_C2, self._target_xyz(1, lifted=True))

        elif self.phase == Phase.SWIRL_C2 and within_settle:
            self._enter(Phase.WAIT, None)

        elif self.phase == Phase.MOVE_C3 and within_settle:
            self._enter(Phase.SWIRL_C3, self._target_xyz(2, lifted=True))

        elif self.phase == Phase.SWIRL_C3 and within_settle:
            self._enter(Phase.WAIT, None)

        elif self.phase == Phase.WAIT:
            # do nothing; wait for /state/active_index to select the next center
            pass


def main():
    rclpy.init()
    node = RoboHandNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
