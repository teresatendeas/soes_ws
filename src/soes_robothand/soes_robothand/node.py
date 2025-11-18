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
    HOME  = 0    # go to init_pos (via linear Cartesian path)
    WAIT  = 1    # idle until /state/active_index changes
    MOVE  = 2    # go to center i (Cartesian target, via linear path)
    SWIRL = 3    # generate spiral about center i


class RoboHandNode(Node):
    """
    4-DOF arm: q = [q1 (yaw), q2, q3, q4] with analytic FK/J.
    - index = -1 -> HOME (drive EE along a line to home)
    - index in {0,1,2} -> MOVE along line to centers[i], then SWIRL around that center
    - Publishes /arm/at_target (Bool) when within tolerance (HOME/MOVE/SWIRL)
    """
    def __init__(self):
        super().__init__('soes_robothand')

        # -------- Control rate & tolerances --------
        self.declare_parameter('rate_hz', 20.0)
        self.declare_parameter('pos_tol_m', 0.003)    # Cartesian tol
        self.declare_parameter('settle_s', 0.20)      # dwell inside tol before declaring "at target"

        # -------- Geometry (L1..L4) --------
        self.declare_parameter('link_lengths_m', [0.00, 0.14, 0.12, 0.04])  # [L1,L2,L3,L4]

        # --------  tuning --------
        self.declare_parameter('kp_cart', 3.0)
        self.declare_parameter('damping_lambda', 0.1)
        self.declare_parameter('qdot_limit_rad_s', [1.5, 1.5, 1.5, 1.5])
        self.declare_parameter('q_min_rad', [-math.pi, -math.pi/2, -math.pi/2, -math.pi/2])
        self.declare_parameter('q_max_rad', [ math.pi,  math.pi/2,  math.pi/2,  math.pi/2])

        # -------- HOME (joint space definition) --------
        self.declare_parameter('q_home_rad', [0.0, 0.0, 0.0, 0.0])  # safe ready pose in joint space
        self.declare_parameter('kp_joint', 3.0)                     # (kept for compatibility)
        self.declare_parameter('home_tol_rad', 0.02)                # ~1.1° (kept for compatibility)

        # -------- Spiral parameters --------
        # r(θ) = R0 * (1 + α θ), z(θ) = (height / θ_max) * θ,  θ̇ = ω
        self.declare_parameter('R0', 0.025)
        self.declare_parameter('turns', 3)
        self.declare_parameter('alpha', -0.03)
        self.declare_parameter('height', 0.04)
        self.declare_parameter('omega', 0.5)  # rad/s

        # -------- far-move slow mode --------
        self.declare_parameter('far_distance_m', 0.15)      # "far" threshold in Cartesian space
        self.declare_parameter('far_home_err_rad', 1.0)     # kept for YAML compatibility (not used directly)
        self.declare_parameter('far_speed_scale', 0.4)      # 0 < scale <= 1, e.g. 0.4 => 40% of normal speed

        # -------- NEW: linear-path speed (in Cartesian space) --------
        # This controls how fast we move along the line p(s) = a + b*s.
        self.declare_parameter('line_speed_m_s', 0.05)  # 5 cm/s along the line

        # -------- Load parameters --------
        self.rate_hz  = float(self.get_parameter('rate_hz').value)
        self.dt       = 1.0 / self.rate_hz
        self.pos_tol  = float(self.get_parameter('pos_tol_m').value)
        self.settle_s = float(self.get_parameter('settle_s').value)

        self.links    = np.array(self.get_parameter('link_lengths_m').value, dtype=float)
        self.L1, self.L2, self.L3, self.L4 = map(float, self.links)

        self.kp       = float(self.get_parameter('kp_cart').value)
        self.lmbda    = float(self.get_parameter('damping_lambda').value)
        self.qdot_lim = np.array(self.get_parameter('qdot_limit_rad_s').value, dtype=float)
        self.q_min    = np.array(self.get_parameter('q_min_rad').value, dtype=float)
        self.q_max    = np.array(self.get_parameter('q_max_rad').value, dtype=float)

        self.q_home   = np.array(self.get_parameter('q_home_rad').value, dtype=float)
        self.kp_joint = float(self.get_parameter('kp_joint').value)   # unused now, but kept
        self.home_tol = float(self.get_parameter('home_tol_rad').value)

        self.R0     = float(self.get_parameter('R0').value)
        self.turns  = int(self.get_parameter('turns').value)
        self.alpha  = float(self.get_parameter('alpha').value)
        self.height = float(self.get_parameter('height').value)
        self.omega  = float(self.get_parameter('omega').value)
        self.theta_max = 2.0 * math.pi * self.turns
        self.s = (self.height / self.theta_max) if self.theta_max != 0.0 else 0.0

        # far/slow params
        self.far_distance_m   = float(self.get_parameter('far_distance_m').value)
        self.far_home_err_rad = float(self.get_parameter('far_home_err_rad').value)  # not used directly
        self.far_speed_scale  = float(self.get_parameter('far_speed_scale').value)
        self.qdot_lim_slow    = self.far_speed_scale * self.qdot_lim

        # linear path speed
        self.line_speed = float(self.get_parameter('line_speed_m_s').value)

        # -------- ROS I/O --------
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.index_sub   = self.create_subscription(Int32, '/state/active_index', self._on_index, 10)
        self.center_sub  = self.create_subscription(CupcakeCenters, '/vision/centers', self._on_centers, qos)
        self.targets_pub = self.create_publisher(JointTargets, '/arm/joint_targets', 10)
        self.at_pub      = self.create_publisher(Bool, '/arm/at_target', 1)
        self.swirl_pub   = self.create_publisher(Bool, '/arm/swirl_active', 1)   # used by StateNode

        # pause input
        self.paused = False
        self.create_subscription(Bool, '/esp_paused', self._on_paused, 10)

        # -------- Runtime --------
        self.q: np.ndarray = np.zeros(4, dtype=float)  # current joints
        self.active_index: int = -1
        self.centers: Optional[List[Tuple[float,float,float]]] = None

        self.phase = Phase.HOME
        self.phase_t0 = self.get_clock().now()
        self.last_within_tol = None
        self.des_xyz: Optional[np.ndarray] = None

        # Spiral bookkeeping
        self.spiral_theta = 0.0
        self.spiral_center: Optional[np.ndarray] = None

        # Logging helpers
        self._home_done_logged = False

        # -------- Linear path state: p(s) = a + b*s, s in [0,1] --------
        self.line_active: bool = False
        self.line_a: Optional[np.ndarray] = None
        self.line_b: Optional[np.ndarray] = None
        self.line_s: float = 0.0
        self.line_dir: float = 1.0  # 1 forward, -1 backward

        # (optional) path logging if you still want it
        self.path_logging: bool = False
        self.path_points: List[np.ndarray] = []

        self.timer = self.create_timer(self.dt, self._tick)
        self.get_logger().info('soes_robothand: HOME first, then MOVE/SWIRL on index.')

    # ------------- Callbacks -------------
    def _on_index(self, msg: Int32):
        prev = self.active_index
        self.active_index = int(msg.data)
        self.get_logger().info(f"active_index: {prev} -> {self.active_index}")
        self._align_phase_with_index()

    def _on_centers(self, msg: CupcakeCenters):
        self.centers = [(p.x, p.y, p.z) for p in msg.centers]
        if len(self.centers) < 3:
            self.get_logger().warn("centers < 3; waiting for all three targets")

    def _on_paused(self, msg: Bool):
        """Freeze arm control loop when ESP pause is active."""
        self.paused = bool(msg.data)

    # ------------- Phase selection -------------
    def _align_phase_with_index(self):
        # Go to home: use linear Cartesian path from current EE pose to FK(q_home)
        if self.active_index == -1:
            home_xyz = self.fk_xyz(self.q_home)
            self.get_logger().info("[ROBOHAND] Moving to init pos (HOME) along linear Cartesian path")
            self._enter(Phase.HOME, home_xyz)

        # Move to one of the cupcake positions (pos1, pos2, pos3) via linear path
        elif self.active_index in (0, 1, 2) and self.centers and len(self.centers) >= 3:
            label = f"pos{self.active_index + 1}"
            target_xyz = np.array(self.centers[self.active_index], dtype=float)
            self.get_logger().info(f"[ROBOHAND] Moving to {label} along linear Cartesian path")
            self._enter(Phase.MOVE, target_xyz)
        else:
            self._enter(Phase.WAIT, None)

    def _enter(self, new_phase: Phase, xyz: Optional[np.ndarray]):
        self.phase = new_phase
        self.phase_t0 = self.get_clock().now()
        self.last_within_tol = None
        self.des_xyz = xyz.copy() if xyz is not None else None

        # Reset path logging by default
        self.path_logging = False
        self.path_points = []

        # For HOME and MOVE, set up a linear path: p(s) = a + b*s
        if new_phase in (Phase.HOME, Phase.MOVE) and xyz is not None:
            self._setup_line_to(xyz, forward=True)
            self.path_logging = True  # optional: record path actually executed
            if new_phase == Phase.HOME:
                self.get_logger().info("[LINE] HOME: linear path from current EE to home")
            else:
                self.get_logger().info("[LINE] MOVE: linear path from current EE to cupcake center")
        else:
            # For WAIT / SWIRL, disable line tracking
            self.line_active = False
            self.line_a = None
            self.line_b = None
            self.line_s = 0.0

        # Reset HOME arrival logging when entering HOME
        if new_phase == Phase.HOME:
            self._home_done_logged = False

        self.get_logger().info(f"[ROBOHAND] -> {self.phase.name} des={self.des_xyz if xyz is not None else None}")

    def _elapsed(self) -> float:
        return (self.get_clock().now() - self.phase_t0).nanoseconds * 1e-9

    # ------------- Linear path helpers -------------
    def _setup_line_to(self, target_xyz: np.ndarray, forward: bool = True):
        """Set up a new line p(s) = a + b*s from current EE position to target_xyz."""
        start_xyz = self.fk_xyz(self.q)
        self.line_a = start_xyz.copy()
        self.line_b = target_xyz - start_xyz
        self.line_dir = 1.0 if forward else -1.0
        self.line_s = 0.0 if forward else 1.0
        self.line_active = True

        self.get_logger().info(
            "[LINE] a=(%.4f, %.4f, %.4f), target=(%.4f, %.4f, %.4f)" %
            (self.line_a[0], self.line_a[1], self.line_a[2],
             target_xyz[0], target_xyz[1], target_xyz[2])
        )

    def _line_step(self) -> Tuple[np.ndarray, bool]:
        """
        Advance along the line by one time-step.
        Returns (des_xyz, done) where done=True when s reaches the end.
        """
        if not self.line_active or self.line_a is None or self.line_b is None:
            # Fallback: just stay at des_xyz
            if self.des_xyz is not None:
                return self.des_xyz.copy(), True
            return self.fk_xyz(self.q), True

        b_norm = float(np.linalg.norm(self.line_b))
        if b_norm < 1e-6:
            # start and target are almost identical
            self.line_s = 1.0
            des = self.line_a + self.line_b
            return des, True

        # speed along the line (|b| * ds/dt = line_speed)
        ds = (self.line_speed * self.dt) / b_norm
        self.line_s += self.line_dir * ds

        done = False
        if self.line_dir > 0.0:
            if self.line_s >= 1.0:
                self.line_s = 1.0
                done = True
        else:
            if self.line_s <= 0.0:
                self.line_s = 0.0
                done = True

        des = self.line_a + self.line_b * self.line_s
        return des, done

    # ------------- Analytic FK & J (your model) -------------
    def fk_xyz(self, q: np.ndarray) -> np.ndarray:
        q1, q2, q3, q4 = q
        L1, L2, L3, L4 = self.L1, self.L2, self.L3, self.L4
        r_fk = L2*math.cos(q2) + L3*math.cos(q2+q3) + L4*math.cos(q2+q3+q4)
        z_fk = L1 + L2*math.sin(q2) + L3*math.sin(q2+q3) + L4*math.sin(q2+q3+q4)
        x_fk = r_fk * math.cos(q1)
        y_fk = r_fk * math.sin(q1)
        return np.array([x_fk, y_fk, z_fk], dtype=float)

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        q1, q2, q3, q4 = q
        L2, L3, L4 = self.L2, self.L3, self.L4
        r_fk   = L2*math.cos(q2) + L3*math.cos(q2+q3) + L4*math.cos(q2+q3+q4)
        dr_dq2 = -L2*math.sin(q2) - L3*math.sin(q2+q3) - L4*math.sin(q2+q3+q4)
        dr_dq3 = -L3*math.sin(q2+q3) - L4*math.sin(q2+q3+q4)
        dr_dq4 = -L4*math.sin(q2+q3+q4)
        dz_dq2 =  L2*math.cos(q2) + L3*math.cos(q2+q3) + L4*math.cos(q2+q3+q4)
        dz_dq3 =  L3*math.cos(q2+q3) + L4*math.cos(q2+q3+q4)
        dz_dq4 =  L4*math.cos(q2+q3+q4)

        J = np.zeros((3,4))
        J[:,0] = [-r_fk*math.sin(q1), r_fk*math.cos(q1), 0.0]
        J[:,1] = [math.cos(q1)*dr_dq2, math.sin(q1)*dr_dq2, dz_dq2]
        J[:,2] = [math.cos(q1)*dr_dq3, math.sin(q1)*dr_dq3, dz_dq3]
        J[:,3] = [math.cos(q1)*dr_dq4, math.sin(q1)*dr_dq4, dz_dq4]
        return J

    # ------------- Controllers -------------
    def _publish_targets(self, q: np.ndarray, qdot: np.ndarray, use_velocity: bool):
        msg = JointTargets()
        msg.position = [float(a) for a in q]
        msg.velocity = [float(w) for w in qdot]
        msg.use_velocity = bool(use_velocity)
        self.targets_pub.publish(msg)

    def _publish_at(self, is_at: bool):
        self.at_pub.publish(Bool(data=bool(is_at)))

    def _publish_swirl(self, active: bool):
        """Tell StateNode whether we are in SWIRL phase or not."""
        self.swirl_pub.publish(Bool(data=bool(active)))

    def _ik_step(self, des_xyz: np.ndarray, xdot_ff: Optional[np.ndarray] = None) -> bool:
        cur_xyz = self.fk_xyz(self.q)

        # optional path logging for debugging
        if self.path_logging:
            self.path_points.append(cur_xyz.copy())

        err = des_xyz - cur_xyz
        dist = float(np.linalg.norm(err))

        if dist <= self.pos_tol:
            if self.last_within_tol is None:
                self.last_within_tol = self.get_clock().now()
        else:
            self.last_within_tol = None

        v = self.kp * err
        if xdot_ff is not None:
            v = v + xdot_ff

        J = self.jacobian(self.q)
        JJt = J @ J.T
        qdot = J.T @ np.linalg.solve(JJt + (self.lmbda**2) * np.eye(3), v)

        # choose joint velocity limits: slow when "far" in Cartesian space"
        if dist >= self.far_distance_m:
            qdot_lim = self.qdot_lim_slow
        else:
            qdot_lim = self.qdot_lim

        qdot = np.clip(qdot, -qdot_lim, qdot_lim)
        self.q = np.clip(self.q + qdot * self.dt, self.q_min, self.q_max)

        self._publish_targets(self.q, qdot, use_velocity=True)

        at = (
            self.last_within_tol is not None and
            (self.get_clock().now() - self.last_within_tol) >= Duration(seconds=self.settle_s)
        )

        self._publish_at(at)
        return at

    # ------------- Phase logic -------------
    def _start_swirl(self):
        # prepare spiral about current center
        if self.centers is None or self.active_index not in (0,1,2):
            return

        label = f"pos{self.active_index + 1}"
        self.get_logger().info(f"[ROBOHAND] Arrived at {label}, starting swirl")

        self.spiral_center = np.array(self.centers[self.active_index], dtype=float)
        self.spiral_theta = 0.0
        self._enter(Phase.SWIRL, self.spiral_center.copy())

    def _tick(self):
        # ======== GLOBAL PAUSE GATING ========
        if self.paused:
            # Do not update q, spiral_theta, or publish new commands while paused.
            return

        # HOME: follow linear path from current EE position to home
        if self.phase == Phase.HOME and self.des_xyz is not None:
            des, done_line = self._line_step()
            at = self._ik_step(des)
            self._publish_swirl(False)

            # optional log when we reach home
            if done_line and at and not self._home_done_logged:
                self.get_logger().info("[ROBOHAND] Arrived at init pos (HOME) via linear path")
                self._home_done_logged = True
            return

        # WAIT
        if self.phase == Phase.WAIT:
            self._publish_at(False)
            self._publish_swirl(False)
            return

        # MOVE: follow linear path from init_pos (or current) to cupcake center
        if self.phase == Phase.MOVE and self.des_xyz is not None:
            des, done_line = self._line_step()
            at = self._ik_step(des)
            if done_line and at:
                # at the "start of swirl" (cupcake center): now start SWIRL
                self._start_swirl()
            self._publish_swirl(False)
            return

        # SWIRL
        if self.phase == Phase.SWIRL and self.spiral_center is not None:
            if self.theta_max <= 0.0:
                self._enter(Phase.WAIT, None)
                self._publish_swirl(False)
                return

            # Spiral pose around the center
            r = self.R0 * (1.0 + self.alpha * self.spiral_theta)
            dx = r * math.cos(self.spiral_theta)
            dy = r * math.sin(self.spiral_theta)
            dz = self.s * self.spiral_theta   # linear height with theta
            des = self.spiral_center + np.array([dx, dy, dz])

            # Spiral feedforward (helps tracking)
            rdot = self.R0 * self.alpha * self.omega
            xdot = rdot * math.cos(self.spiral_theta) - r * self.omega * math.sin(self.spiral_theta)
            ydot = rdot * math.sin(self.spiral_theta) + r * self.omega * math.cos(self.spiral_theta)
            zdot = self.s * self.omega
            ff = np.array([xdot, ydot, zdot])

            self._ik_step(des, ff)
            self.spiral_theta += self.omega * self.dt

            if self.spiral_theta >= self.theta_max:
                label = f"pos{self.active_index + 1}" if self.active_index in (0,1,2) else "current position"
                self.get_logger().info(f"[SWIRL] Swirl done at {label}")
                self._enter(Phase.WAIT, None)
                self._publish_swirl(False)   # SWIRL ended
            else:
                self.get_logger().debug("[SWIRL] Active")
                self._publish_swirl(True)    # still swirling
            return

        # default
        self._publish_at(False)
        self._publish_swirl(False)


def main():
    rclpy.init()
    node = RoboHandNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
