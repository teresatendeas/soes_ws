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
    HOME  = 0    # joint-space home (index = -1)
    WAIT  = 1    # idle until /state/active_index changes
    MOVE  = 2    # go to center i (Cartesian target)
    SWIRL = 3    # generate spiral about center i


class RoboHandNode(Node):
    """
    4-DOF arm: q = [q1 (yaw), q2, q3, q4] with analytic FK/J.
    - index = -1 -> HOME (drive joints to q_home_rad)
    - index in {0,1,2} -> MOVE to centers[i], then SWIRL a spiral about that center
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

        # -------- HOME (joint space) --------
        self.declare_parameter('q_home_rad', [0.0, 0.0, 0.0, 0.0])  # set this to a safe ready pose
        self.declare_parameter('kp_joint', 3.0)                     # joint homing gain
        self.declare_parameter('home_tol_rad', 0.02)                # ~1.1°

        # -------- Spiral parameters --------
        # r(θ) = R0 * (1 + α θ), z(θ) = (height / θ_max) * θ,  θ̇ = ω
        self.declare_parameter('R0', 0.025)
        self.declare_parameter('turns', 3)
        self.declare_parameter('alpha', -0.03)
        self.declare_parameter('height', 0.04)
        self.declare_parameter('omega', 0.5)  # rad/s

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
        self.kp_joint = float(self.get_parameter('kp_joint').value)
        self.home_tol = float(self.get_parameter('home_tol_rad').value)

        self.R0     = float(self.get_parameter('R0').value)
        self.turns  = int(self.get_parameter('turns').value)
        self.alpha  = float(self.get_parameter('alpha').value)
        self.height = float(self.get_parameter('height').value)
        self.omega  = float(self.get_parameter('omega').value)
        self.theta_max = 2.0 * math.pi * self.turns
        self.s = (self.height / self.theta_max) if self.theta_max != 0.0 else 0.0

        # -------- ROS I/O --------
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.index_sub   = self.create_subscription(Int32, '/state/active_index', self._on_index, 10)
        self.center_sub  = self.create_subscription(CupcakeCenters, '/vision/centers', self._on_centers, qos)
        self.targets_pub = self.create_publisher(JointTargets, '/arm/joint_targets', 10)
        self.at_pub      = self.create_publisher(Bool, '/arm/at_target', 1)
        self.swirl_pub   = self.create_publisher(Bool, '/arm/swirl_active', 1)   # NEW

        # -------- Runtime --------
        self.q: np.ndarray = np.zeros(4, dtype=float)
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
        self._home_done_logged = False  # ensure "arrived HOME" logged once

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

    # ------------- Phase selection -------------
    def _align_phase_with_index(self):
        # Moving to init pos (HOME)
        if self.active_index == -1:
            self.get_logger().info("[ROBOHAND] Moving to init pos (HOME)")
            self._enter(Phase.HOME, None)
        # Moving to one of the cupcake positions
        elif self.active_index in (0, 1, 2) and self.centers and len(self.centers) >= 3:
            label = f"pos{self.active_index + 1}"
            self.get_logger().info(f"[ROBOHAND] Moving to {label}")
            self._enter(Phase.MOVE, np.array(self.centers[self.active_index], dtype=float))
        else:
            self._enter(Phase.WAIT, None)

    def _enter(self, new_phase: Phase, xyz: Optional[np.ndarray]):
        self.phase = new_phase
        self.phase_t0 = self.get_clock().now()
        self.last_within_tol = None
        self.des_xyz = xyz.copy() if xyz is not None else None

        # Reset HOME arrival logging when entering HOME
        if new_phase == Phase.HOME:
            self._home_done_logged = False

        self.get_logger().info(f"[ROBOHAND] -> {self.phase.name} des={self.des_xyz if xyz is not None else None}")

    def _elapsed(self) -> float:
        return (self.get_clock().now() - self.phase_t0).nanoseconds * 1e-9

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

    def _publish_swirl(self, active: bool):          # NEW
        """Tell StateNode whether we are in SWIRL phase or not."""  # NEW
        self.swirl_pub.publish(Bool(data=bool(active)))            # NEW

    def _home_step(self) -> bool:
        """Joint-space home control."""
        err = self.q_home - self.q
        qdot = self.kp_joint * err
        qdot = np.clip(qdot, -self.qdot_lim, self.qdot_lim)
        self.q = np.clip(self.q + qdot * self.dt, self.q_min, self.q_max)

        self._publish_targets(self.q, np.zeros(4), use_velocity=False)
        at = float(np.linalg.norm(err)) <= self.home_tol
        self._publish_at(at)

        # Log once when HOME reached
        if at and not self._home_done_logged:
            self.get_logger().info("[ROBOHAND] Arrived at init pos (HOME)")
            self._home_done_logged = True

        return at

    def _ik_step(self, des_xyz: np.ndarray, xdot_ff: Optional[np.ndarray] = None) -> bool:
        cur_xyz = self.fk_xyz(self.q)
        err = des_xyz - cur_xyz
        if np.linalg.norm(err) <= self.pos_tol:
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
        qdot = np.clip(qdot, -self.qdot_lim, self.qdot_lim)
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
        # HOME
        if self.phase == Phase.HOME:
            self._home_step()
            self._publish_swirl(False)   # NEW
            return

        # WAIT
        if self.phase == Phase.WAIT:
            self._publish_at(False)
            self._publish_swirl(False)   # NEW
            return

        # MOVE
        if self.phase == Phase.MOVE and self.des_xyz is not None:
            at = self._ik_step(self.des_xyz)
            if at:
                self._start_swirl()
            self._publish_swirl(False)   # NEW
            return

        # SWIRL
        if self.phase == Phase.SWIRL and self.spiral_center is not None:
            if self.theta_max <= 0.0:
                self._enter(Phase.WAIT, None)
                self._publish_swirl(False)   # NEW
                return

            # Spiral pose
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
                self._publish_swirl(False)   # NEW: SWIRL ended
            else:
                self.get_logger().debug("[SWIRL] Active")
                self._publish_swirl(True)    # NEW: still swirling
            return

        # default
        self._publish_at(False)
        self._publish_swirl(False)           # NEW


def main():
    rclpy.init()
    node = RoboHandNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
