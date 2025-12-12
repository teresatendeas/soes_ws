#!/usr/bin/env python3
import enum, math
from typing import Optional, List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration

from std_msgs.msg import Bool, Int32
from soes_msgs.msg import PumpCmd, JointTargets, RollerCmd, CupcakeCenters

from .utils import PumpController


# ---------------- High-level phases (sequence) ----------------
class Phase(enum.Enum):
    INIT_POS    = 0
    STEP0       = 1
    STEP1       = 2
    STEP2       = 3
    CAMERA      = 4
    ROLL_TRAY   = 5
    IDLE        = 6
    TEST_MOTOR  = 7
    POST_STEP   = 8


# ---------------- Arm phases (internal IK) ----------------
class ArmPhase(enum.Enum):
    HOME     = 0   # go to q_home
    WAIT     = 1   # idle until commanded
    MOVE     = 2   # move toward target center
    SWIRL    = 3   # swirl trajectory
    RETRACT  = 4   # move away after swirl (to home)
    HOLD     = 5   # stay still (publish hold)


class StateNode(Node):
    def __init__(self):
        super().__init__('soes_state')

        # ==========  HIGH-LEVEL STATE PARAMETERS  ============
        self.declare_parameter('order', [0, 1, 2])
        self.declare_parameter('roller_distance_mm', 70.0)
        self.declare_parameter('roller_speed_mm_s', 17.5)
        self.declare_parameter('camera_timeout_s', 50.0)

        self.order      = list(self.get_parameter('order').value)
        self.roll_dist  = float(self.get_parameter('roller_distance_mm').value)
        self.roll_speed = float(self.get_parameter('roller_speed_mm_s').value)
        self.cam_to     = float(self.get_parameter('camera_timeout_s').value)

        # ==========   ARM / IK CONTROL PARAMETERS   ==========
        self.declare_parameter('rate_hz', 20.0)
        self.declare_parameter('pos_tol_m', 0.01)
        self.declare_parameter('settle_s', 0.5)
        self.declare_parameter('link_lengths_m', [0.00, 0.17, 0.14, 0.04])
        self.declare_parameter('kp_cart', 1.0)
        self.declare_parameter('damping_lambda', 0.1)
        self.declare_parameter('qdot_limit_rad_s', [3.0, 3.0, 3.0, 3.0])
        self.declare_parameter('q_min_rad', [-314.16, -157.08, -157.08, -1.5708])
        self.declare_parameter('q_max_rad', [ 314.16,  157.08,  157.08,  1.5708])

        # Spiral parameters
        self.declare_parameter('R0', 0.025)
        self.declare_parameter('turns', 3)
        self.declare_parameter('alpha', -0.03)
        self.declare_parameter('height', 0.04)
        self.declare_parameter('omega', 2.0)

        # ----- Load arm params -----
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

        self.R0       = float(self.get_parameter('R0').value)
        self.turns    = int(self.get_parameter('turns').value)
        self.alpha    = float(self.get_parameter('alpha').value)
        self.height   = float(self.get_parameter('height').value)
        self.omega    = float(self.get_parameter('omega').value)

        self.theta_max = 2.0 * math.pi * self.turns
        self.s         = (self.height / self.theta_max) if self.theta_max != 0.0 else 0.0

        # ========== Home pose (INIT_POS) ==========
        self.q_home = np.array([0.0, math.radians(90.0), math.radians(-90.0), 0.0], dtype=float)

        # ==================   ROS I/O   ======================
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )

        self.pump_pub    = self.create_publisher(PumpCmd, '/pump/cmd', 1)
        self.roller_pub  = self.create_publisher(RollerCmd, '/roller/cmd', 1)
        self.arm_pub     = self.create_publisher(JointTargets, '/arm/joint_targets', 10)

        # MISSING pubs in your snippet (needed by your own functions)
        self.phase_pub   = self.create_publisher(Int32, '/soes/phase', 10)
        self.swirl_pub   = self.create_publisher(Bool, '/arm/swirl_active', 10)
        self.arm_at_pub  = self.create_publisher(Bool, '/arm/at_target', 10)

        self.switch_on = True
        self.create_subscription(Bool, '/esp_switch_on', self._on_switch, 10)

        self.paused = False
        self.pause_start = None
        self.create_subscription(Bool, '/esp_paused', self._on_paused, 10)

        self.vision_request_pub = self.create_publisher(Bool, '/vision/request', 1)
        self.vision_done: Optional[bool] = None
        self.create_subscription(Bool, '/vision/soes_done', self._on_vision_done, 10)

        self.create_subscription(CupcakeCenters, '/vision/centers', self._on_centers, qos)

        self.pump = PumpController(self._pump_on, self._pump_off)

        # ===================   RUNTIME   =====================
        self.phase = Phase.IDLE
        self.phase_t0 = self.get_clock().now()
        self._did_start_pump = False

        self._roller_active = False
        self._roller_duration_s = 0.0

        # Arm runtime
        self.q: np.ndarray = np.zeros(4, dtype=float)
        self._last_qdot_cmd = np.zeros(4, dtype=float)

        self.arm_phase = ArmPhase.WAIT
        self.arm_phase_t0 = self.get_clock().now()
        self.last_within_tol = None
        self.des_xyz: Optional[np.ndarray] = None

        self.spiral_theta = 0.0
        self.spiral_center: Optional[np.ndarray] = None

        self.swirl_active = False

        # Vision centers + step control
        self.centers: Optional[List[Tuple[float, float, float]]] = None
        self.step_idx = 0                # 0..2 for STEP0..STEP2
        self.target_idx = 0              # 0..2 from order[]
        self._home_done_logged = False

        # Tick timer (your code needs this to run)
        self.timer = self.create_timer(self.dt, self._on_timer)

        self.get_logger().info('soes_state: ready (IDLE, waiting RESET LOW).')
        self._publish_phase()

    # ==================  TIMER ===========================
    def _on_timer(self):
        # keep pump controller alive if it has internal timing
        if hasattr(self.pump, "tick"):
            try:
                self.pump.tick(self.dt)
            except Exception as e:
                self.get_logger().warn(f"PumpController.tick error: {e}")

        self.tick()
        self._arm_tick()

    # ==================  VISION CALLBACKS  ===============
    def _on_vision_done(self, msg: Bool):
        self.vision_done = bool(msg.data)
        self.get_logger().info(f'Received soes_done = {self.vision_done}')

    def _on_centers(self, msg: CupcakeCenters):
        self.centers = [(p.x, p.y, p.z) for p in msg.centers]

        if len(self.centers) < 3:
            self.get_logger().warn("centers < 3; waiting for all three targets")
            return

    # ==================  GENERIC HELPERS  ================
    def _enter(self, new_phase: Phase):
        old_phase = self.phase
        self.phase = new_phase
        self.phase_t0 = self.get_clock().now()
        self._did_start_pump = False

        if new_phase != Phase.ROLL_TRAY:
            self._roller_active = False
            self._roller_duration_s = 0.0

        if new_phase == Phase.CAMERA:
            self.vision_done = None
            req = Bool()
            req.data = True
            self.vision_request_pub.publish(req)
            self.get_logger().info('CAMERA: sent /vision/request = True, waiting /vision/soes_done')

        # Hook arm behavior to overall phase
        if new_phase == Phase.INIT_POS:
            self._arm_enter(ArmPhase.HOME, None)

        if new_phase in (Phase.STEP0, Phase.STEP1, Phase.STEP2):
            self._setup_step_target()

        self.get_logger().warn(f'[OVERALL] {old_phase.name} → {new_phase.name}')
        self.get_logger().info(f'[STATE] -> {self.phase.name}')
        self._publish_phase()

    def _elapsed(self) -> float:
        return (self.get_clock().now() - self.phase_t0).nanoseconds * 1e-9

    def _publish_phase(self):
        msg = Int32()
        msg.data = int(self.phase.value)
        self.phase_pub.publish(msg)

    def _pump_on(self, duty: float, duration_s: float):
        msg = PumpCmd()
        msg.on = True
        msg.duty = float(duty)
        msg.duration_s = float(duration_s)
        self.pump_pub.publish(msg)

    def _pump_off(self):
        msg = PumpCmd()
        msg.on = False
        msg.duty = 0.0
        msg.duration_s = 0.0
        self.pump_pub.publish(msg)

    def _roller_cmd(self, on: bool):
        msg = RollerCmd()
        msg.on = bool(on)
        self.roller_pub.publish(msg)

    def _set_arm_at(self, at: bool):
        self.arm_at_pub.publish(Bool(data=bool(at)))

    # ==================  SWITCH / PAUSE  =================
    def _on_switch(self, msg: Bool):
        prev = self.switch_on
        self.switch_on = bool(msg.data)

        if prev and not self.switch_on:
            self.get_logger().warn('RESET pressed (HIGH -> LOW) -> INIT_POS.')
            self.pump.stop()
            self._roller_cmd(False)
            self._set_swirl_active(False)
            self._set_arm_at(False)
            self.step_idx = 0
            self._enter(Phase.INIT_POS)

        elif (not prev) and self.switch_on:
            self.get_logger().warn('RESET released (LOW -> HIGH) -> IDLE.')
            self.pump.stop()
            self._roller_cmd(False)
            self._enter(Phase.IDLE)

    def _on_paused(self, msg: Bool):
        new_state = bool(msg.data)

        if new_state and not self.paused:
            self.pause_start = self.get_clock().now()

        elif not new_state and self.paused:
            if self.pause_start is not None:
                dt = self.get_clock().now() - self.pause_start
                self.phase_t0 = self.phase_t0 + dt
                self.arm_phase_t0 = self.arm_phase_t0 + dt
                self.pause_start = None

        self.paused = new_state

    # ==================  MAIN TICK (STATE) ===============
    def tick(self):
        if self.paused:
            return

        if self.phase == Phase.INIT_POS:
            # Arm motion handled by _arm_tick HOME.
            # When home reached, start next step.
            if self.arm_phase == ArmPhase.HOLD:
                # Decide next phase based on step_idx
                if self.step_idx == 0:
                    self.get_logger().info("INIT_POS -> STEP0")
                    self._enter(Phase.STEP0)
                elif self.step_idx == 1:
                    self.get_logger().info("INIT_POS -> STEP1")
                    self._enter(Phase.STEP1)
                elif self.step_idx == 2:
                    self.get_logger().info("INIT_POS -> STEP2")
                    self._enter(Phase.STEP2)
                else:
                    self.get_logger().info("All steps done -> CAMERA")
                    self._enter(Phase.CAMERA)

        elif self.phase == Phase.STEP0:
            if self._run_step():
                self.get_logger().info("STEP0 complete -> INIT_POS")
                self.step_idx = 1
                self._enter(Phase.INIT_POS)

        elif self.phase == Phase.STEP1:
            if self._run_step():
                self.get_logger().info("STEP1 complete -> INIT_POS")
                self.step_idx = 2
                self._enter(Phase.INIT_POS)

        elif self.phase == Phase.STEP2:
            if self._run_step():
                self.get_logger().info("STEP2 done -> INIT_POS before CAMERA")
                self.step_idx = 3
                self._enter(Phase.INIT_POS)

        elif self.phase == Phase.CAMERA:
            if self.vision_done is not None:
                self.get_logger().info("Vision is done")
                if self.vision_done:
                    self.get_logger().info("Vision OK → ROLL_TRAY")
                else:
                    self.get_logger().warn("Vision BAD → ROLL_TRAY (tetap jalan tray)")
                self._enter(Phase.ROLL_TRAY)
                return

            if self._elapsed() >= self.cam_to:
                self.get_logger().warn("Camera timeout → ROLL_TRAY")
                self._enter(Phase.ROLL_TRAY)
                return

        elif self.phase == Phase.ROLL_TRAY:
            roll_time = self.roll_dist / max(self.roll_speed, 1e-3)
            t = self._elapsed()

            if not self._roller_active:
                self._roller_active = True
                self._roller_duration_s = roll_time
                self._roller_cmd(True)
                self.get_logger().info(f'ROLLER ON for {roll_time:.2f} s')
                return

            if t >= self._roller_duration_s:
                self._roller_cmd(False)
                self._roller_active = False
                self.get_logger().info('ROLLER OFF, restart.')
                # restart cycle
                self.step_idx = 0
                self._enter(Phase.INIT_POS)

        elif self.phase == Phase.IDLE:
            # The robot arm should NOT move here
            # Ensure it holds still
            self._hold_position()
            return

    # ==================  STEP LOGIC  =====================
    def _start_step(self, step_idx: int):
        self._enter(
            Phase.STEP0 if step_idx == 0
            else Phase.STEP1 if step_idx == 1
            else Phase.STEP2
        )

    def _run_step(self) -> bool:
        # Step ends when swirl finished (pump was running only during swirl_active)
        if self.swirl_active:
            if not self._did_start_pump:
                self.pump.start(duty=1.0, duration_s=0.0)
                self._did_start_pump = True
                self.get_logger().info('Pump ON (SWIRL)')
            return False

        if self._did_start_pump:
            self.pump.stop()
            self._did_start_pump = False
            self.get_logger().info('Pump OFF (step complete)')
            return True

        return False

    # ==================  ARM / IK PART  ==================
    def _set_swirl_active(self, active: bool):
        self.swirl_active = bool(active)
        self.swirl_pub.publish(Bool(data=self.swirl_active))

    def _arm_enter(self, new_phase: ArmPhase, xyz: Optional[np.ndarray]):
        old_phase = self.arm_phase
        self.arm_phase = new_phase
        self.arm_phase_t0 = self.get_clock().now()
        self.last_within_tol = None
        self.des_xyz = xyz.copy() if xyz is not None else None

        if new_phase == ArmPhase.HOME:
            self._home_done_logged = False

        if new_phase != old_phase:
            if self.des_xyz is not None:
                self.get_logger().info(
                    f"[ROBOHAND] Phase {old_phase.name} -> {new_phase.name}, des={self.des_xyz}"
                )
            else:
                self.get_logger().info(
                    f"[ROBOHAND] Phase {old_phase.name} -> {new_phase.name}"
                )

    def _arm_elapsed(self) -> float:
        return (self.get_clock().now() - self.arm_phase_t0).nanoseconds * 1e-9

    # ---------- FK & Jacobian ----------
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

        J = np.zeros((3, 4))
        J[:, 0] = [-r_fk*math.sin(q1), r_fk*math.cos(q1), 0.0]
        J[:, 1] = [math.cos(q1)*dr_dq2, math.sin(q1)*dr_dq2, dz_dq2]
        J[:, 2] = [math.cos(q1)*dr_dq3, math.sin(q1)*dr_dq3, dz_dq3]
        J[:, 3] = [math.cos(q1)*dr_dq4, math.sin(q1)*dr_dq4, dz_dq4]
        return J

    def _publish_targets(self, q: np.ndarray, qdot: np.ndarray, use_velocity: bool):
        msg = JointTargets()
        msg.position = [float(a) for a in q]
        msg.velocity = [float(w) for w in qdot]
        msg.use_velocity = bool(use_velocity)
        self.arm_pub.publish(msg)

    def _hold_position(self):
        qdot0 = np.zeros(4, dtype=float)
        self._publish_targets(self.q, qdot0, use_velocity=False)

    def _home_step(self, speed_scale: float = 1.0) -> bool:
        # Joint-space approach to q_home (stable and minimal dependencies)
        err = self.q_home - self.q
        if np.linalg.norm(err) < 1e-6:
            self._hold_position()
            return True

        # simple P in joint space
        k = 3.0
        qdot = k * err
        limit = self.qdot_lim * speed_scale
        qdot = np.clip(qdot, -limit, limit)

        self.q = np.clip(self.q + qdot * self.dt, self.q_min, self.q_max)
        self._last_qdot_cmd = qdot.copy()
        self._publish_targets(self.q, qdot, use_velocity=True)

        close = np.max(np.abs(self.q_home - self.q)) < math.radians(1.0)
        if close:
            self._hold_position()
        return close

    def _ik_step(
        self,
        des_xyz: np.ndarray,
        xdot_ff: Optional[np.ndarray] = None,
        speed_scale: float = 1.0
    ) -> bool:
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

        limit = self.qdot_lim * speed_scale
        qdot = np.clip(qdot, -limit, limit)

        self.q = np.clip(self.q + qdot * self.dt, self.q_min, self.q_max)

        self._last_qdot_cmd = qdot.copy()
        self._publish_targets(self.q, qdot, use_velocity=True)

        at = (
            self.last_within_tol is not None and
            (self.get_clock().now() - self.last_within_tol) >= Duration(seconds=self.settle_s)
        )

        self._set_arm_at(at)
        return at

    def _setup_step_target(self):
        # Choose target based on order list
        if not self.order:
            self.get_logger().warn("[ROBOHAND] order empty, cannot run step")
            return

        if self.step_idx > 2:
            return

        # map step_idx -> order index safely
        oi = min(self.step_idx, len(self.order) - 1)
        self.target_idx = int(self.order[oi])
        self.target_idx = 0 if self.target_idx < 0 else 2 if self.target_idx > 2 else self.target_idx

        if self.centers is None or len(self.centers) < 3:
            self.get_logger().warn("[ROBOHAND] centers not ready, staying WAIT")
            self._arm_enter(ArmPhase.WAIT, None)
            return

        self.spiral_center = np.array(self.centers[self.target_idx], dtype=float)
        self.spiral_theta = 0.0

        label = f"pos{self.target_idx + 1}"
        self.get_logger().info(f"[ROBOHAND] STEP{self.step_idx}: target={label}, center={self.spiral_center}")

        # Start by moving to center (MOVE). Swirl starts after at-target.
        self._arm_enter(ArmPhase.MOVE, self.spiral_center.copy())

    def _start_swirl(self):
        if self.spiral_center is None:
            return

        label = f"pos{self.target_idx + 1}"
        self.get_logger().info(f"[ROBOHAND] Arrived at {label}, starting swirl")
        self.spiral_theta = 0.0
        self._arm_enter(ArmPhase.SWIRL, self.spiral_center.copy())

    def _go_home(self):
        self._arm_enter(ArmPhase.RETRACT, None)

    def _arm_tick(self):
        # I want 4 phases: Move towards swirl, Making swirl, Move away, staying still
        if self.paused:
            return

        # In IDLE, do not move
        if self.phase == Phase.IDLE:
            self._set_swirl_active(False)
            self._set_arm_at(False)
            self._hold_position()
            self.arm_phase = ArmPhase.HOLD
            return

        # HOME: go to q_home and then HOLD
        if self.arm_phase == ArmPhase.HOME:
            at = self._home_step(speed_scale=0.6)
            if at:
                if not self._home_done_logged:
                    self.get_logger().info("[ROBOHAND] Arrived at INIT_POS (q_home). HOLD")
                    self._home_done_logged = True
                self._set_swirl_active(False)
                self._set_arm_at(True)
                self.arm_phase = ArmPhase.HOLD
                self._hold_position()
            return

        # HOLD: keep still
        if self.arm_phase == ArmPhase.HOLD:
            self._set_swirl_active(False)
            self._hold_position()
            return

        # WAIT: centers not ready or not commanded
        if self.arm_phase == ArmPhase.WAIT:
            self._set_swirl_active(False)
            self._set_arm_at(False)
            self._hold_position()
            return

        # MOVE: go to center, then SWIRL
        if self.arm_phase == ArmPhase.MOVE and self.des_xyz is not None:
            at = self._ik_step(self.des_xyz, xdot_ff=None, speed_scale=0.9)
            self._set_swirl_active(False)
            if at:
                self._start_swirl()
            return

        # SWIRL
        if self.arm_phase == ArmPhase.SWIRL and self.spiral_center is not None:
            if self.theta_max <= 0.0:
                self._set_swirl_active(False)
                self._go_home()
                return

            r = self.R0 * (1.0 + self.alpha * self.spiral_theta)
            dx = r * math.cos(self.spiral_theta)
            dy = r * math.sin(self.spiral_theta)
            dz = self.s * self.spiral_theta
            des = self.spiral_center + np.array([dx, dy, dz])

            rdot = self.R0 * self.alpha * self.omega
            xdot = rdot * math.cos(self.spiral_theta) - r * self.omega * math.sin(self.spiral_theta)
            ydot = rdot * math.sin(self.spiral_theta) + r * self.omega * math.cos(self.spiral_theta)
            zdot = self.s * self.omega
            ff = np.array([xdot, ydot, zdot])

            self._ik_step(des, ff, speed_scale=1.0)
            self.spiral_theta += self.omega * self.dt

            if self.spiral_theta >= self.theta_max:
                label = f"pos{self.target_idx + 1}" if self.target_idx in (0, 1, 2) else "target"
                self.get_logger().info(f"[SWIRL] Swirl done at {label}")
                self._set_swirl_active(False)
                self._go_home()
            else:
                self._set_swirl_active(True)
            return

        # RETRACT: go back to home, then HOLD
        if self.arm_phase == ArmPhase.RETRACT:
            at = self._home_step(speed_scale=0.8)
            self._set_swirl_active(False)
            if at:
                self.get_logger().info("[ROBOHAND] Retract done. HOLD")
                self._set_arm_at(True)
                self.arm_phase = ArmPhase.HOLD
                self._hold_position()
            return

        # fallback
        self._set_arm_at(False)
        self._set_swirl_active(False)
        self._hold_position()


def main():
    rclpy.init()
    node = StateNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
