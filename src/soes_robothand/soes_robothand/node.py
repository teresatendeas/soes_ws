#!/usr/bin/env python3
import enum, math
from typing import Optional, Tuple, List

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Bool, Int32
from soes_msgs.msg import JointTargets, CupcakeCenters


# ---------------- Phases ----------------
class Phase(enum.Enum):
    HOME  = 0
    WAIT  = 1
    MOVE  = 2
    SWIRL = 3


class RoboHandNode(Node):
    """
    4-DOF arm: q = [yaw, q2, q3, q4].
    Uses analytic FK & J, damped-least-squares IK.
    Publishes /arm/at_target and /arm/swirl_active.
    """

    def __init__(self):
        super().__init__('soes_robothand')

        # -------- Parameters --------
        self.declare_parameter('rate_hz', 20.0)
        self.declare_parameter('pos_tol_m', 0.003)
        self.declare_parameter('settle_s', 0.20)

        self.declare_parameter('link_lengths_m', [0.00, 0.14, 0.12, 0.04])

        self.declare_parameter('kp_cart', 3.0)
        self.declare_parameter('damping_lambda', 0.1)
        self.declare_parameter('qdot_limit_rad_s', [1.5, 1.5, 1.5, 1.5])
        self.declare_parameter('q_min_rad', [-math.pi, -math.pi/2, -math.pi/2, -math.pi/2])
        self.declare_parameter('q_max_rad', [ math.pi,  math.pi/2,  math.pi/2,  math.pi/2])

        self.declare_parameter('q_home_rad', [0.0, 0.0, 0.0, 0.0])
        self.declare_parameter('kp_joint', 3.0)
        self.declare_parameter('home_tol_rad', 0.02)

        self.declare_parameter('R0', 0.025)
        self.declare_parameter('turns', 3)
        self.declare_parameter('alpha', -0.03)
        self.declare_parameter('height', 0.04)
        self.declare_parameter('omega', 0.5)

        # -------- Load parameters --------
        self.rate_hz  = float(self.get_parameter('rate_hz').value)
        self.dt       = 1.0 / self.rate_hz
        self.pos_tol  = float(self.get_parameter('pos_tol_m').value)
        self.settle_s = float(self.get_parameter('settle_s').value)

        self.links = np.array(self.get_parameter('link_lengths_m').value, dtype=float)
        self.L1, self.L2, self.L3, self.L4 = map(float, self.links)

        self.kp       = float(self.get_parameter('kp_cart').value)
        self.lmbda    = float(self.get_parameter('damping_lambda').value)
        self.qdot_lim = np.array(self.get_parameter('qdot_limit_rad_s').value, dtype=float)
        self.q_min    = np.array(self.get_parameter('q_min_rad').value, dtype=float)
        self.q_max    = np.array(self.get_parameter('q_max_rad').value, dtype=float)

        self.q_home   = np.array(self.get_parameter('q_home_rad').value, dtype=float)
        self.kp_joint = float(self.get_parameter('kp_joint').value)
        self.home_tol = float(self.get_parameter('home_tol_rad').value)

        self.R0     = float(self.get_parameter('R0').value)
        self.turns  = int(self.get_parameter('turns').value)
        self.alpha  = float(self.get_parameter('alpha').value)
        self.height = float(self.get_parameter('height').value)
        self.omega  = float(self.get_parameter('omega').value)

        self.theta_max = 2.0 * math.pi * self.turns
        self.s = (self.height / self.theta_max) if self.theta_max > 0 else 0.0

        # -------- ROS I/O --------
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)

        self.index_sub   = self.create_subscription(Int32, '/state/active_index',
                                                    self._on_index, 10)
        self.center_sub  = self.create_subscription(CupcakeCenters, '/vision/centers',
                                                    self._on_centers, qos)

        self.targets_pub = self.create_publisher(JointTargets, '/arm/joint_targets', 10)
        self.at_pub      = self.create_publisher(Bool, '/arm/at_target', 1)
        self.swirl_pub   = self.create_publisher(Bool, '/arm/swirl_active', 1)

        # -------- Runtime State --------
        self.q = np.zeros(4, dtype=float)
        self.active_index = -1
        self.centers: Optional[List[Tuple[float, float, float]]] = None

        self.phase = Phase.HOME
        self.phase_t0 = self.get_clock().now()
        self.last_within_tol = None
        self.des_xyz: Optional[np.ndarray] = None

        # Spiral bookkeeping
        self.spiral_theta = 0.0
        self.spiral_center: Optional[np.ndarray] = None

        # Timer
        self.timer = self.create_timer(self.dt, self._tick)

        self.get_logger().info("soes_robothand READY — starting in HOME phase")

    # ============================================================
    # CALLBACKS
    # ============================================================
    def _on_index(self, msg: Int32):
        prev = self.active_index
        self.active_index = int(msg.data)
        self.get_logger().info(f"[INDEX] {prev} → {self.active_index}")
        self._align_phase_with_index()

    def _on_centers(self, msg: CupcakeCenters):
        self.centers = [(p.x, p.y, p.z) for p in msg.centers]
        self.get_logger().info(f"[CENTERS] Received {len(self.centers)} centers")

        if len(self.centers) < 3:
            self.get_logger().warn("[CENTERS] Less than 3 detected — waiting")

    # ============================================================
    # PHASE MANAGEMENT
    # ============================================================
    def _align_phase_with_index(self):
        if self.active_index == -1:
            self._enter(Phase.HOME, None)
        elif self.centers and len(self.centers) >= 3:
            center = np.array(self.centers[self.active_index])
            self._enter(Phase.MOVE, center)
        else:
            self._enter(Phase.WAIT, None)

    def _enter(self, new_phase: Phase, xyz: Optional[np.ndarray]):
        old = self.phase
        self.phase = new_phase
        self.phase_t0 = self.get_clock().now()
        self.des_xyz = xyz.copy() if xyz is not None else None
        self.last_within_tol = None

        self.get_logger().info(
            f"[PHASE] {old.name} → {new_phase.name}  target={self.des_xyz}"
        )

    def _elapsed(self) -> float:
        return (self.get_clock().now() - self.phase_t0).nanoseconds * 1e-9

    # ============================================================
    # FORWARD KINEMATICS & JACOBIAN
    # ============================================================
    def fk_xyz(self, q: np.ndarray) -> np.ndarray:
        q1, q2, q3, q4 = q
        L1, L2, L3, L4 = self.L1, self.L2, self.L3, self.L4

        r = (L2*math.cos(q2) +
             L3*math.cos(q2+q3) +
             L4*math.cos(q2+q3+q4))
        z = (L1 +
             L2*math.sin(q2) +
             L3*math.sin(q2+q3) +
             L4*math.sin(q2+q3+q4))

        x = r * math.cos(q1)
        y = r * math.sin(q1)
        return np.array([x, y, z])

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        q1, q2, q3, q4 = q
        L2, L3, L4 = self.L2, self.L3, self.L4

        r = (L2*math.cos(q2) +
             L3*math.cos(q2+q3) +
             L4*math.cos(q2+q3+q4))
        dr2 = -L2*math.sin(q2) - L3*math.sin(q2+q3) - L4*math.sin(q2+q3+q4)
        dr3 = -L3*math.sin(q2+q3) - L4*math.sin(q2+q3+q4)
        dr4 = -L4*math.sin(q2+q3+q4)

        dz2 = L2*math.cos(q2) + L3*math.cos(q2+q3) + L4*math.cos(q2+q3+q4)
        dz3 = L3*math.cos(q2+q3) + L4*math.cos(q2+q3+q4)
        dz4 = L4*math.cos(q2+q3+q4)

        J = np.zeros((3,4))
        J[:,0] = [-r*math.sin(q1), r*math.cos(q1), 0.0]
        J[:,1] = [math.cos(q1)*dr2, math.sin(q1)*dr2, dz2]
        J[:,2] = [math.cos(q1)*dr3, math.sin(q1)*dr3, dz3]
        J[:,3] = [math.cos(q1)*dr4, math.sin(q1)*dr4, dz4]
        return J

    # ============================================================
    # CONTROLLERS
    # ============================================================
    def _publish_targets(self, q, qdot, vel_mode):
        msg = JointTargets()
        msg.position = list(map(float, q))
        msg.velocity = list(map(float, qdot))
        msg.use_velocity = bool(vel_mode)
        self.targets_pub.publish(msg)

    def _publish_at(self, at: bool):
        self.at_pub.publish(Bool(data=at))

    def _publish_swirl(self, active: bool):
        self.swirl_pub.publish(Bool(data=active))
        self.get_logger().debug(f"[SWIRL_ACTIVE] → {active}")

    # ============================================================
    # HOME CONTROL
    # ============================================================
    def _home_step(self) -> bool:
        err = self.q_home - self.q
        err_norm = np.linalg.norm(err)

        qdot = np.clip(self.kp_joint * err, -self.qdot_lim, self.qdot_lim)
        self.q = np.clip(self.q + qdot*self.dt, self.q_min, self.q_max)

        self._publish_targets(self.q, np.zeros(4), False)

        self.get_logger().debug(f"[HOME] err={err_norm:.4f} q={self.q}")

        at = err_norm <= self.home_tol
        self._publish_at(at)

        if at:
            self.get_logger().info("[HOME] Reached home pose")
        return at

    # ============================================================
    # IK CONTROL
    # ============================================================
    def _ik_step(self, des_xyz, ff=None) -> bool:
        cur = self.fk_xyz(self.q)
        err = des_xyz - cur
        err_norm = np.linalg.norm(err)

        self.get_logger().debug(
            f"[IK] err={err_norm:.4f}m des={des_xyz} cur={cur} q={self.q}"
        )

        # Settling logic
        if err_norm <= self.pos_tol:
            if self.last_within_tol is None:
                self.get_logger().info("[IK] Position reached — settling")
                self.last_within_tol = self.get_clock().now()
        else:
            if self.last_within_tol is not None:
                self.get_logger().warn("[IK] Left tolerance window")
            self.last_within_tol = None

        v = self.kp * err
        if ff is not None:
            v += ff

        J = self.jacobian(self.q)
        JJt = J @ J.T
        qdot = J.T @ np.linalg.solve(JJt + (self.lmbda**2)*np.eye(3), v)
        qdot = np.clip(qdot, -self.qdot_lim, self.qdot_lim)

        self.q = np.clip(self.q + qdot*self.dt, self.q_min, self.q_max)
        self._publish_targets(self.q, qdot, True)

        # Settled?
        at = (
            self.last_within_tol is not None and
            (self.get_clock().now() - self.last_within_tol) >= Duration(seconds=self.settle_s)
        )
        self._publish_at(at)

        if at:
            self.get_logger().info("[IK] IK target settled")
        return at

    # ============================================================
    # SWIRL
    # ============================================================
    def _start_swirl(self):
        if self.centers is None or self.active_index not in (0,1,2):
            self.get_logger().error("[SWIRL] Cannot start swirl — missing centers")
            return

        self.spiral_center = np.array(self.centers[self.active_index])
        self.spiral_theta = 0.0

        self.get_logger().info(f"[SWIRL] Start swirl at {self.spiral_center}")
        self._enter(Phase.SWIRL, self.spiral_center.copy())

    # ============================================================
    # TICK LOOP
    # ============================================================
    def _tick(self):
        # HOME
        if self.phase == Phase.HOME:
            self._home_step()
            self._publish_swirl(False)
            return

        # WAIT
        if self.phase == Phase.WAIT:
            self._publish_at(False)
            self._publish_swirl(False)
            return

        # MOVE
        if self.phase == Phase.MOVE and self.des_xyz is not None:
            at = self._ik_step(self.des_xyz)
            if at:
                self._start_swirl()
            self._publish_swirl(False)
            return

        # SWIRL
        if self.phase == Phase.SWIRL and self.spiral_center is not None:

            if self.theta_max <= 0:
                self.get_logger().error("[SWIRL] Invalid theta_max")
                self._enter(Phase.WAIT, None)
                self._publish_swirl(False)
                return

            # Spiral definition
            r = self.R0 * (1 + self.alpha*self.spiral_theta)
            dx = r * math.cos(self.spiral_theta)
            dy = r * math.sin(self.spiral_theta)
            dz = self.s * self.spiral_theta

            des = self.spiral_center + np.array([dx, dy, dz])

            # Feedforward
            rdot = self.R0 * self.alpha * self.omega
            xdot = rdot*math.cos(self.spiral_theta) - r*self.omega*math.sin(self.spiral_theta)
            ydot = rdot*math.sin(self.spiral_theta) + r*self.omega*math.cos(self.spiral_theta)
            zdot = self.s * self.omega

            ff = np.array([xdot, ydot, zdot])

            # Log spiral progress
            self.get_logger().debug(
                f"[SWIRL] theta={self.spiral_theta:.3f} des={des} q={self.q}"
            )

            self._ik_step(des, ff)
            self.spiral_theta += self.omega*self.dt

            if self.spiral_theta >= self.theta_max:
                self.get_logger().info("[SWIRL] Completed spiral")
                self._enter(Phase.WAIT, None)
                self._publish_swirl(False)
            else:
                self._publish_swirl(True)
            return

        # DEFAULT
        self._publish_at(False)
        self._publish_swirl(False)


def main():
    rclpy.init()
    node = RoboHandNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
