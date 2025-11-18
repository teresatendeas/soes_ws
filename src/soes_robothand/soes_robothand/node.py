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
    SWIRL = 3    # generate spiral about center i (or S-shape in Option A)


class RoboHandNode(Node):
    """
    4-DOF arm: q = [q1 (yaw), q2, q3, q4] with analytic FK/J.
    - index = -1 -> HOME (drive joints to q_home_rad)
    - index in {0,1,2} -> MOVE to centers[i], then SWIRL a spiral about that center
    - Publishes /arm/at_target (Bool) when within tolerance (HOME/MOVE/SWIRL)

    Option A modifications:
    - SWIRL phase replaced by 2-DOF S-shape generator (XY S inside circle at given Z).
    - Only actively move q[0] (yaw) and q[1] (pitch). q[2], q[3] are kept unchanged (zeros or q_home).
    - New ROS parameters (s-shape) added with defaults; can be tuned via YAML/ros2 param.
    """

    def __init__(self):
        super().__init__('soes_robothand')

        # -------- Control rate & tolerances --------
        self.declare_parameter('rate_hz', 20.0)
        self.declare_parameter('pos_tol_m', 0.003)    # Cartesian tol
        self.declare_parameter('settle_s', 0.20)      # dwell inside tol before declaring "at target"

        # -------- Geometry (L1..L4) --------
        self.declare_parameter('link_lengths_m', [0.00, 0.14, 0.12, 0.04])  # [L1,L2,L3,L4]

        # -------- tuning --------
        self.declare_parameter('kp_cart', 3.0)
        self.declare_parameter('damping_lambda', 0.1)
        self.declare_parameter('qdot_limit_rad_s', [1.5, 1.5, 1.5, 1.5])
        self.declare_parameter('q_min_rad', [-math.pi, -math.pi/2, -math.pi/2, -math.pi/2])
        self.declare_parameter('q_max_rad', [ math.pi,  math.pi/2,  math.pi/2,  math.pi/2])

        # -------- HOME (joint space) --------
        self.declare_parameter('q_home_rad', [0.0, 0.0, 0.0, 0.0])  # set this to a safe ready pose
        self.declare_parameter('kp_joint', 3.0)                     # joint homing gain
        self.declare_parameter('home_tol_rad', 0.02)                # ~1.1°

        # -------- Spiral parameters (legacy) --------
        # r(θ) = R0 * (1 + α θ), z(θ) = (height / θ_max) * θ,  θ̇ = ω
        self.declare_parameter('R0', 0.025)
        self.declare_parameter('turns', 3)
        self.declare_parameter('alpha', -0.03)
        self.declare_parameter('height', 0.04)
        self.declare_parameter('omega', 0.5)  # rad/s

        # -------- NEW: S-shape parameters (Option A) --------
        # All distances/lengths here are in meters to match link_lengths_m
        self.declare_parameter('s_shape_R_m', 0.03)        # radius of bounding circle (m)
        self.declare_parameter('s_shape_freq', 2)         # number of wiggles across sweep
        self.declare_parameter('s_shape_amp_scale', 0.95) # amplitude scale (0..1)
        self.declare_parameter('s_shape_n_points', 600)   # number of samples in S-curve
        self.declare_parameter('s_shape_total_time_s', 8.0)  # time to traverse S-curve

        # -------- NEW: S-curve profile times (for HOME and MOVE) --------
        # These control how long the S-curve ramp takes. Tune in YAML.
        self.declare_parameter('move_profile_time_s', 1.0)
        self.declare_parameter('home_profile_time_s', 1.0)

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

        # S-shape params
        self.s_shape_R       = float(self.get_parameter('s_shape_R_m').value)
        self.s_shape_freq    = int(self.get_parameter('s_shape_freq').value)
        self.s_shape_amp     = float(self.get_parameter('s_shape_amp_scale').value)
        self.s_shape_n       = int(self.get_parameter('s_shape_n_points').value)
        self.s_shape_total_t = float(self.get_parameter('s_shape_total_time_s').value)

        # NEW: profile durations
        self.move_T = float(self.get_parameter('move_profile_time_s').value)
        self.home_T = float(self.get_parameter('home_profile_time_s').value)

        # -------- ROS I/O --------
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.index_sub   = self.create_subscription(Int32, '/state/active_index', self._on_index, 10)
        self.center_sub  = self.create_subscription(CupcakeCenters, '/vision/centers', self._on_centers, qos)
        self.targets_pub = self.create_publisher(JointTargets, '/arm/joint_targets', 10)
        self.at_pub      = self.create_publisher(Bool, '/arm/at_target', 1)
        self.swirl_pub   = self.create_publisher(Bool, '/arm/swirl_active', 1)   # already used by StateNode

        # NEW: subscribe to /esp_paused to freeze this node too
        self.paused = False
        self.create_subscription(Bool, '/esp_paused', self._on_paused, 10)

        # -------- Runtime --------
        # keep q as length 4 to remain compatible with downstream interfaces
        self.q: np.ndarray = np.array(self.q_home.copy(), dtype=float)
        self.active_index: int = -1
        self.centers: Optional[List[Tuple[float,float,float]]] = None

        self.phase = Phase.HOME
        self.phase_t0 = self.get_clock().now()
        self.last_within_tol = None
        self.des_xyz: Optional[np.ndarray] = None

        # S-shape bookkeeping (Option A)
        self.s_path: Optional[np.ndarray] = None  # Nx3 (m)
        self.s_tvec: Optional[np.ndarray] = None
        self.s_idx: int = 0
        self.s_total_time: float = 0.0

        # Spiral bookkeeping (legacy, still present)
        self.spiral_theta = 0.0
        self.spiral_center: Optional[np.ndarray] = None

        # Active DOFs mask: only q0,q1 actively commanded by S-shape IK
        # Keep q2,q3 fixed (hold near q_home)
        self.active_dofs_mask = np.array([True, True, False, False], dtype=bool)

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

    def _on_paused(self, msg: Bool):
        """Freeze arm control loop when ESP pause is active."""
        self.paused = bool(msg.data)

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

    # ------------- NEW: S-curve speed scaling -------------
    def _s_curve_speed(self, profile_T: float) -> float:
        """
        Smooth S-curve-like speed factor in [0.1, 1].
        Uses the derivative of a smoothstep (4*tau*(1-tau)) as a bell-shaped profile.
        """
        if profile_T <= 0.0:
            return 1.0
        t = self._elapsed()
        tau = max(0.0, min(t / profile_T, 1.0))  # 0..1
        # Bell-shaped curve: 0 at start/end, 1 at middle
        scale = 4.0 * tau * (1.0 - tau)
        # Avoid fully zero -> keep at least 0.1 so the arm still moves
        return max(scale, 0.1)

    # ------------- S-shape generator (meters) -------------
    def _generate_s_shape_for_center(
        self,
        center: np.ndarray,
        R: float = None,
        n_points: Optional[int] = None,
        freq: Optional[int] = None,
        amp_scale: Optional[float] = None,
        z_draw: Optional[float] = None,
        total_time: Optional[float] = None
    ):
        """
        Generate an S-shaped path inside a circle around the given center.
        Inputs/outputs are in meters (consistent with node link lengths).
        center: np.array([Cx, Cy, Cz]) in meters
        Returns dict with keys: Xd, Yd, Zd, Xd_feas, Yd_feas, Zd_feas, yaw, pitch, EE_x, EE_y, EE_z, t_vec, clamped_count
        """
        Cx, Cy, Cz = float(center[0]), float(center[1]), float(center[2])
        R = self.s_shape_R if R is None else float(R)
        n_points = self.s_shape_n if n_points is None else int(n_points)
        freq = self.s_shape_freq if freq is None else int(freq)
        amp_scale = self.s_shape_amp if amp_scale is None else float(amp_scale)
        total_time = self.s_shape_total_t if total_time is None else float(total_time)
        # default z_draw: use center's z if not provided
        z_draw = Cz if z_draw is None else float(z_draw)

        # Reachability sanity: using meters
        if z_draw < (self.L1 - self.L2) or z_draw > (self.L1 + self.L2):
            raise ValueError('z_draw outside reachable vertical band. Choose other z_draw.')

        # Generate S curve (in meters)
        y_sweep = np.linspace(Cy - R, Cy + R, n_points)
        Xd = np.zeros(n_points)
        Yd = np.zeros(n_points)
        Zd = np.full(n_points, z_draw)

        for k in range(n_points):
            yk = y_sweep[k]
            half_w = math.sqrt(max(0.0, R**2 - (yk - Cy)**2))
            amp = amp_scale * half_w
            u = k / (n_points - 1)
            xk = Cx + amp * math.sin(2.0 * math.pi * freq * u)
            Xd[k] = xk
            Yd[k] = yk

        # Reachability check & clamping (meters)
        tol = 1e-9
        Xd_feas = np.zeros_like(Xd)
        Yd_feas = np.zeros_like(Yd)
        Zd_feas = Zd.copy()

        clamped_count = 0
        for i in range(len(Xd)):
            px = Xd[i]; py = Yd[i]; pz = Zd[i]
            dxy = math.hypot(px, py)
            dz = pz - self.L1
            sdist = math.sqrt(dxy*dxy + dz*dz)
            if sdist <= self.L2 + tol:
                Xd_feas[i] = px; Yd_feas[i] = py
            else:
                clamped_count += 1
                scale = self.L2 / sdist
                dxy_c = dxy * scale
                dz_c = dz * scale
                if dxy > tol:
                    Xd_feas[i] = (px / dxy) * dxy_c
                    Yd_feas[i] = (py / dxy) * dxy_c
                else:
                    Xd_feas[i] = 0.0
                    Yd_feas[i] = 0.0
                Zd_feas[i] = self.L1 + dz_c

        # Inverse kinematics (yaw + pitch) in radians for verification (not used directly to command joints here)
        yaw = np.arctan2(Yd_feas, Xd_feas)
        dproj = np.hypot(Xd_feas, Yd_feas)
        pitch = np.arctan2(Zd_feas - self.L1, dproj)
        near_zero = dproj < 1e-9
        pitch = np.where(near_zero & (Zd_feas > self.L1), +math.pi/2.0, pitch)
        pitch = np.where(near_zero & (Zd_feas < self.L1), -math.pi/2.0, pitch)

        # Forward kinematics (verify)
        EE_x = self.L2 * np.cos(pitch) * np.cos(yaw)
        EE_y = self.L2 * np.cos(pitch) * np.sin(yaw)
        EE_z = self.L1 + self.L2 * np.sin(pitch)

        # Time vector
        t_vec = np.linspace(0.0, total_time, len(Xd))

        return {
            "Xd": Xd, "Yd": Yd, "Zd": Zd,
            "Xd_feas": Xd_feas, "Yd_feas": Yd_feas, "Zd_feas": Zd_feas,
            "yaw": yaw, "pitch": pitch, "EE_x": EE_x, "EE_y": EE_y, "EE_z": EE_z,
            "t_vec": t_vec, "clamped_count": clamped_count
        }

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
        # Ensure we always publish 4 entries (some controllers expect fixed length)
        pos = list(q)
        vel = list(qdot)
        # pad if necessary
        while len(pos) < 4:
            pos.append(float(self.q_home[len(pos)] if len(self.q_home) > len(pos) else 0.0))
        while len(vel) < 4:
            vel.append(0.0)
        msg = JointTargets()
        msg.position = [float(a) for a in pos[:4]]
        msg.velocity = [float(w) for w in vel[:4]]
        msg.use_velocity = bool(use_velocity)
        self.targets_pub.publish(msg)

    def _publish_at(self, is_at: bool):
        self.at_pub.publish(Bool(data=bool(is_at)))

    def _publish_swirl(self, active: bool):
        """Tell StateNode whether we are in SWIRL phase or not."""
        self.swirl_pub.publish(Bool(data=bool(active)))

    def _home_step(self, speed_scale: float = 1.0) -> bool:
        """Joint-space home control with S-curve speed scaling."""
        err = self.q_home - self.q
        qdot = self.kp_joint * err

        # Apply S-curve speed scaling to joint velocity limit
        limit = self.qdot_lim * speed_scale
        qdot = np.clip(qdot, -limit, limit)
        self.q = np.clip(self.q + qdot * self.dt, self.q_min, self.q_max)

        self._publish_targets(self.q, np.zeros(4), use_velocity=False)
        at = float(np.linalg.norm(err)) <= self.home_tol
        self._publish_at(at)

        # Log once when HOME reached
        if at and not self._home_done_logged:
            self.get_logger().info("[ROBOHAND] Arrived at init pos (HOME)")
            self._home_done_logged = True

        return at

    def _ik_step(
        self,
        des_xyz: np.ndarray,
        xdot_ff: Optional[np.ndarray] = None,
        speed_scale: float = 1.0
    ) -> bool:
        """Cartesian IK step with S-curve speed scaling on joint velocity limits.
        Modified for Option A: only update joints marked in self.active_dofs_mask.
        """
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

        # zero out qdot for inactive DOFs (Option A: keep q2,q3 at home)
        if hasattr(self, 'active_dofs_mask') and len(self.active_dofs_mask) == len(qdot):
            qdot = np.where(self.active_dofs_mask, qdot, 0.0)
        else:
            # fallback: if mask length mismatch, keep behavior unchanged
            pass

        # Apply S-curve speed scaling to joint velocity limits
        limit = self.qdot_lim * speed_scale
        qdot = np.clip(qdot, -limit, limit)
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
        # prepare S-shape about current center (Option A)
        if self.centers is None or self.active_index not in (0,1,2):
            return

        label = f"pos{self.active_index + 1}"
        self.get_logger().info(f"[ROBOHAND] Arrived at {label}, starting S-shape")

        center = np.array(self.centers[self.active_index], dtype=float)
        # generate S-shape in meters (center is assumed in meters)
        res = self._generate_s_shape_for_center(
            center=center,
            R=self.s_shape_R,
            n_points=self.s_shape_n,
            freq=self.s_shape_freq,
            amp_scale=self.s_shape_amp,
            z_draw=center[2],
            total_time=self.s_shape_total_t
        )

        # store feasible path (meters)
        Xf = res['Xd_feas']; Yf = res['Yd_feas']; Zf = res['Zd_feas']
        self.s_path = np.vstack((Xf, Yf, Zf)).T  # Nx3
        self.s_tvec = res['t_vec']
        self.s_total_time = self.s_shape_total_t
        self.s_idx = 0
        self.spiral_center = center.copy()  # reuse variable name for compatibility
        self.spiral_theta = 0.0

        self._enter(Phase.SWIRL, self.spiral_center.copy())

    def _tick(self):
        # ======== GLOBAL PAUSE GATING ========
        if self.paused:
            # Do not update q, spiral_theta, or publish new commands while paused.
            return

        # HOME
        if self.phase == Phase.HOME:
            # S-curve on the way back to HOME
            speed_scale = self._s_curve_speed(self.home_T)
            self._home_step(speed_scale=speed_scale)
            self._publish_swirl(False)
            return

        # WAIT
        if self.phase == Phase.WAIT:
            self._publish_at(False)
            self._publish_swirl(False)
            return

        # MOVE
        if self.phase == Phase.MOVE and self.des_xyz is not None:
            # S-curve for move from init_pos -> swirl start
            speed_scale = self._s_curve_speed(self.move_T)
            at = self._ik_step(self.des_xyz, speed_scale=speed_scale)
            if at:
                self._start_swirl()
            self._publish_swirl(False)
            return

        # SWIRL (Option A: follow precomputed S-shape path)
        if self.phase == Phase.SWIRL and self.s_path is not None:
            # time-based index into path
            elapsed = self._elapsed()
            if self.s_total_time <= 0.0:
                # nothing to do: end immediately
                self._enter(Phase.WAIT, None)
                self._publish_swirl(False)
                return

            frac = min(1.0, elapsed / self.s_total_time)
            idx = int(frac * (self.s_path.shape[0] - 1))
            des = self.s_path[idx]

            # compute feedforward velocity (finite difference)
            if idx < self.s_path.shape[0] - 1:
                dt = self.s_tvec[1] - self.s_tvec[0] if self.s_tvec.size > 1 else self.dt
                next_p = self.s_path[min(self.s_path.shape[0]-1, idx+1)]
                ff = (next_p - des) / max(dt, 1e-9)
            else:
                ff = np.zeros(3)

            # call IK step (but only joints in active mask will update)
            self._ik_step(des, xdot_ff=ff, speed_scale=1.0)

            if frac >= 1.0:
                label = f"pos{self.active_index + 1}" if self.active_index in (0,1,2) else "current position"
                self.get_logger().info(f"[S-SHAPE] Done at {label}")
                self._enter(Phase.WAIT, None)
                self._publish_swirl(False)
            else:
                self.get_logger().debug("[S-SHAPE] Active")
                self._publish_swirl(True)
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
