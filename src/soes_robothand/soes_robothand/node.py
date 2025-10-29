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
    WAIT  = 0
    MOVE  = 1
    SWIRL = 2


class RoboHandNode(Node):
    """
    Resolved-rate IK (differential kinematics) for a 4-DOF arm:
      q = [yaw, pitch1, pitch2, pitch3]

    - Base yaw rotates a planar 3-link chain (pitch joints).
    - FK is simple geometric; Jacobian is numerical (finite differences).
    - Controller: damped least-squares resolved rate:
        qdot = J^T (J J^T + λ^2 I)^(-1) * (Kp * (x_des - x))
    - Behavior:
        /state/active_index = {0,1,2}  -> MOVE to that center
        settle within tolerance        -> SWIRL (+lift in Z)
        settle again                   -> WAIT for next index
    """
    def __init__(self):
        super().__init__('soes_robothand')

        # ---------- Parameters ----------
        self.declare_parameter('rate_hz', 50.0)
        self.declare_parameter('pos_tol_m', 0.003)          # 3 mm
        self.declare_parameter('settle_s', 0.20)            # dwell time within tol
        self.declare_parameter('lift_height_m', 0.03)       # +3 cm for SWIRL

        # Link lengths [m] for the planar chain (after base yaw)
        self.declare_parameter('link_lengths_m', [0.12, 0.12, 0.10])  # [L1, L2, L3]

        # IK tuning
        self.declare_parameter('kp_cart', 2.0)              # task-space gain [1/s]
        self.declare_parameter('damping_lambda', 0.01)      # DLS damping
        self.declare_parameter('qdot_limit_rad_s', [1.5, 1.5, 1.5, 1.5])
        self.declare_parameter('q_min_rad', [-math.pi, -math.pi/2, -math.pi/2, -math.pi/2])
        self.declare_parameter('q_max_rad', [ math.pi,  math.pi/2,  math.pi/2,  math.pi/2])

        self.rate_hz  = float(self.get_parameter('rate_hz').value)
        self.dt       = 1.0 / self.rate_hz
        self.pos_tol  = float(self.get_parameter('pos_tol_m').value)
        self.settle_s = float(self.get_parameter('settle_s').value)
        self.lift_h   = float(self.get_parameter('lift_height_m').value)

        self.links    = np.array(self.get_parameter('link_lengths_m').value, dtype=float)
        self.kp       = float(self.get_parameter('kp_cart').value)
        self.lmbda    = float(self.get_parameter('damping_lambda').value)
        self.qdot_lim = np.array(self.get_parameter('qdot_limit_rad_s').value, dtype=float)
        self.q_min    = np.array(self.get_parameter('q_min_rad').value, dtype=float)
        self.q_max    = np.array(self.get_parameter('q_max_rad').value, dtype=float)

        # ---------- ROS I/O ----------
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.index_sub   = self.create_subscription(Int32, '/state/active_index', self._on_index, 10)
        self.center_sub  = self.create_subscription(CupcakeCenters, '/vision/centers', self._on_centers, qos)
        self.targets_pub = self.create_publisher(JointTargets, '/arm/joint_targets', 10)

        # ---------- Runtime ----------
        self.q = np.zeros(4, dtype=float)   # joint estimate (replace with feedback when available)
        self.active_index: int = -1
        self.centers: Optional[list[Tuple[float,float,float]]] = None

        self.phase = Phase.WAIT
        self.phase_t0 = self.get_clock().now()
        self.last_within_tol = None
        self.des_xyz: Optional[np.ndarray] = None

        # timer
        self.timer = self.create_timer(self.dt, self._tick)
        self.get_logger().info('soes_robothand (resolved-rate IK) started (generic MOVE/SWIRL).')

    # ================= Subscriptions =================
    def _on_index(self, msg: Int32):
        prev = self.active_index
        self.active_index = int(msg.data)
        self.get_logger().info(f"active_index: {prev} -> {self.active_index}")
        self._on_index_changed()

    def _on_centers(self, msg: CupcakeCenters):
        self.centers = [(p.x, p.y, p.z) for p in msg.centers]
        if len(self.centers) < 3:
            self.get_logger().warn("centers < 3; waiting for full three targets")
            return
        # If an index is already selected and we were waiting for centers, kick off MOVE now.
        if self.phase == Phase.WAIT and self.active_index in (0,1,2):
            self._start_move_to_current_index()

    # ================= Phase / targets =================
    def _on_index_changed(self):
        if self.active_index in (0,1,2) and self.centers and len(self.centers) >= 3:
            self._start_move_to_current_index()
        else:
            # -1 or no centers yet: just wait
            self._enter(Phase.WAIT, None)

    def _start_move_to_current_index(self):
        xyz = self._target_xyz(self.active_index, lifted=False)
        self._enter(Phase.MOVE, xyz)

    def _start_swirl_at_current_index(self):
        xyz = self._target_xyz(self.active_index, lifted=True)
        self._enter(Phase.SWIRL, xyz)

    def _enter(self, new_phase: Phase, xyz: Optional[Tuple[float,float,float]]):
        self.phase = new_phase
        self.phase_t0 = self.get_clock().now()
        self.last_within_tol = None
        self.des_xyz = np.array(xyz, dtype=float) if xyz is not None else None
        self.get_logger().info(f"[ROBOHAND] -> {self.phase.name}  des={self.des_xyz if xyz is not None else None}")

    def _elapsed(self) -> float:
        return (self.get_clock().now() - self.phase_t0).nanoseconds * 1e-9

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
        q = [yaw, th1, th2, th3]
        Base yaw rotates a planar X'Z chain into world XY.
        """
        yaw, th1, th2, th3 = q
        L1, L2, L3 = self.links
        x_p = L1*math.cos(th1) + L2*math.cos(th1+th2) + L3*math.cos(th1+th2+th3)
        z   = L1*math.sin(th1) + L2*math.sin(th1+th2) + L3*math.sin(th1+th2+th3)
        c, s = math.cos(yaw), math.sin(yaw)
        x = c * x_p
        y = s * x_p
        return np.array([x, y, z], dtype=float)

    def jacobian_num(self, q: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        f0 = self.fk_xyz(q)
        J = np.zeros((3, 4), dtype=float)
        for i in range(4):
            dq = np.zeros_like(q); dq[i] = eps
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

        v = self.kp * err
        J = self.jacobian_num(self.q)
        JJt = J @ J.T
        I3 = np.eye(3)
        qdot = J.T @ np.linalg.solve(JJt + (self.lmbda**2) * I3, v)
        qdot = np.clip(qdot, -self.qdot_lim, self.qdot_lim)

        self.q = np.clip(self.q + qdot * self.dt, self.q_min, self.q_max)

        out = JointTargets()
        out.position = [float(a) for a in self.q]
        out.velocity = [float(w) for w in qdot]
        out.use_velocity = True
        self.targets_pub.publish(out)


    # ================= Main loop =================
    def _tick(self):
        if self.centers is None:
            return

        # Drive IK
        if self.des_xyz is not None:
            self.ik_step(self.des_xyz)

        # Transition when we've been within tolerance long enough
        within_settle = (
            self.last_within_tol is not None and
            (self.get_clock().now() - self.last_within_tol) >= Duration(seconds=self.settle_s)
        )

        if self.phase == Phase.MOVE and within_settle:
            # At center -> lift +3 cm
            self._start_swirl_at_current_index()

        elif self.phase == Phase.SWIRL and within_settle:
            # Finished lift -> wait for next index from soes_state
            self._enter(Phase.WAIT, None)

        elif self.phase == Phase.WAIT:
            # sit idle; new /state/active_index will kick off MOVE
            pass


def main():
    rclpy.init()
    node = RoboHandNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
