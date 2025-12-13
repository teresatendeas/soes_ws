#!/usr/bin/env python3
import enum, math
from typing import Optional, List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration

from std_msgs.msg import Bool, Int32
from soes_msgs.msg import PumpCmd, VisionQuality, JointTargets, RollerCmd, CupcakeCenters

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
    HOME  = 0    # joint-space home (index = -1)
    WAIT  = 1    # idle until active_index changes
    MOVE  = 2    # go to center i (Cartesian target)
    SWIRL = 3    # generate spiral about center i


class StateNode(Node):
    def __init__(self):
        super().__init__('soes_state')

        # =====================================================
        # ==========  HIGH-LEVEL STATE PARAMETERS  ============
        # =====================================================
        self.declare_parameter('settle_before_pump_s', 2.0)
        self.declare_parameter('pump_on_s', 2.0)
        self.declare_parameter('swirl_time_s', 1.0)
        self.declare_parameter('order', [0, 1, 2])

        self.declare_parameter('roller_distance_mm', 70.0)
        self.declare_parameter('roller_speed_mm_s', 17.5)

        # camera timeout
        self.declare_parameter('camera_timeout_s', 50.0)

        self.t_settle   = float(self.get_parameter('settle_before_pump_s').value)
        self.t_pump     = float(self.get_parameter('pump_on_s').value)
        self.t_swirl    = float(self.get_parameter('swirl_time_s').value)
        self.order      = list(self.get_parameter('order').value)
        self.roll_dist  = float(self.get_parameter('roller_distance_mm').value)
        self.roll_speed = float(self.get_parameter('roller_speed_mm_s').value)
        self.cam_to     = float(self.get_parameter('camera_timeout_s').value)

        # TEST_MOTOR params
        self.declare_parameter('test_period_s', 3.0)
        self.declare_parameter('test_amp_rad', [3.14, 3.14, 3.14])
        self.declare_parameter('test_servo_deg', [30.0, 150.0])
        self.test_period_s  = float(self.get_parameter('test_period_s').value)
        self.test_amp_rad   = list(self.get_parameter('test_amp_rad').value)
        self.test_servo_deg = list(self.get_parameter('test_servo_deg').value)

        # =====================================================
        # ==========   ARM / IK CONTROL PARAMETERS   ==========
        # =====================================================
        self.declare_parameter('rate_hz', 20.0)
        self.declare_parameter('pos_tol_m', 0.01)
        self.declare_parameter('settle_s', 0.5)

        self.declare_parameter('link_lengths_m', [0.00, 0.17, 0.14, 0.04])

        self.declare_parameter('kp_cart', 1.0)
        self.declare_parameter('damping_lambda', 0.1)
        self.declare_parameter('qdot_limit_rad_s', [3.0, 3.0, 3.0, 3.0])

        self.declare_parameter('q_min_rad', [-314.16, -157.08, -157.08, -1.5708])
        self.declare_parameter('q_max_rad', [ 314.16,  157.08,  157.08,  1.5708])

        self.declare_parameter('q_home_rad', [0.0, 1.5708, -1.5708, 0.0])
        self.declare_parameter('kp_joint', 3.0)
        self.declare_parameter('home_tol_rad', 0.314)

        # Spiral parameters
        self.declare_parameter('R0', 0.025)
        self.declare_parameter('turns', 3)
        self.declare_parameter('alpha', -0.03)
        self.declare_parameter('height', 0.04)
        self.declare_parameter('omega', 2.0)

        # S-curve profile times
        self.declare_parameter('move_profile_time_s', 1.0)
        self.declare_parameter('home_profile_time_s', 1.0)

        # Brake parameters (soft stop)
        self.declare_parameter('brake_time_s', 0.35)

        # Velocity-stability gate (anti-cetek)
        self.declare_parameter('vel_eps_rad_s', 0.20)
        self.declare_parameter('stable_cycles', 5)

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

        self.q_home   = np.array(self.get_parameter('q_home_rad').value, dtype=float)
        self.kp_joint = float(self.get_parameter('kp_joint').value)
        self.home_tol = float(self.get_parameter('home_tol_rad').value)

        self.R0       = float(self.get_parameter('R0').value)
        self.turns    = int(self.get_parameter('turns').value)
        self.alpha    = float(self.get_parameter('alpha').value)
        self.height   = float(self.get_parameter('height').value)
        self.omega    = float(self.get_parameter('omega').value)

        self.theta_max = 2.0 * math.pi * self.turns
        self.s         = (self.height / self.theta_max) if self.theta_max != 0.0 else 0.0

        self.move_T = float(self.get_parameter('move_profile_time_s').value)
        self.home_T = float(self.get_parameter('home_profile_time_s').value)

        self.brake_T = float(self.get_parameter('brake_time_s').value)

        self.vel_eps = float(self.get_parameter('vel_eps_rad_s').value)
        self.stable_cycles = int(self.get_parameter('stable_cycles').value)

        # =====================================================
        # ==================   ROS I/O   ======================
        # =====================================================
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )

        # State / roller / pump / vision quality
        self.index_pub   = self.create_publisher(Int32, '/state/active_index', 1)
        self.pump_pub    = self.create_publisher(PumpCmd, '/pump/cmd', 1)
        self.roller_pub  = self.create_publisher(RollerCmd, '/roller/cmd', 1)
        self.qual_sub    = self.create_subscription(
            VisionQuality, '/vision/quality', self.on_quality, qos
        )

        # Arm command and feedback topics
        self.arm_pub     = self.create_publisher(JointTargets, '/arm/joint_targets', 10)
        self.arm_at_pub  = self.create_publisher(Bool, '/arm/at_target', 1)
        self.swirl_pub   = self.create_publisher(Bool, '/arm/swirl_active', 1)

        # publish state phase
        self.phase_pub = self.create_publisher(Int32, '/state/phase', 1)

        # switch / pause
        self.switch_on = True   # HIGH at startup
        self.create_subscription(Bool, '/esp_switch_on', self._on_switch, 10)

        self.paused = False
        self.pause_start = None
        self.create_subscription(Bool, '/esp_paused', self._on_paused, 10)

        # Vision
        self.vision_request_pub = self.create_publisher(Bool, '/vision/request', 1)
        self.vision_done: Optional[bool] = None
        self.create_subscription(Bool, '/vision/soes_done', self._on_vision_done, 10)
        self.center_sub = self.create_subscription(
            CupcakeCenters, '/vision/centers', self._on_centers, qos
        )

        # Pump helper
        self.pump = PumpController(self._pump_on, self._pump_off)

        # =====================================================
        # ===================   RUNTIME   =====================
        # =====================================================
        # High-level state
        self.phase = Phase.IDLE
        self.phase_t0 = self.get_clock().now()
        self.quality_flag = False
        self._step_idx = 0
        self._did_start_pump = False

        # Roller state
        self._roller_active = False
        self._roller_duration_s = 0.0

        # Arm / IK runtime
        self.q: np.ndarray = np.zeros(4, dtype=float)
        self.active_index: int = -1
        self.centers: Optional[List[Tuple[float, float, float]]] = None

        self.arm_phase = ArmPhase.HOME
        self.arm_phase_t0 = self.get_clock().now()
        self.last_within_tol = None
        self.des_xyz: Optional[np.ndarray] = None

        # Spiral bookkeeping
        self.spiral_theta = 0.0
        self.spiral_center: Optional[np.ndarray] = None

        # Arrival logging
        self._home_done_logged = False

        # Arm at-target + swirl flags (now internal)
        self.arm_at = False
        self.arm_at_since = None
        self.swirl_active = False

        # Braking state (soft stop)
        self._braking = False
        self._brake_t0 = None
        self._brake_qdot0 = np.zeros(4, dtype=float)

        # Stability gate (anti-cetek)
        self._stable_count = 0

        # NEW: HOME hold latch (prevents re-sending tiny velocities when active_index=-1 repeats)
        self._home_hold = False

        # Timers
        self.timer_state = self.create_timer(0.05, self.tick)          # 20 Hz high-level
        self.timer_arm   = self.create_timer(self.dt, self._arm_tick)  # IK loop

        self.get_logger().info('soes_state: ready (IDLE, waiting RESET LOW).')
        self._publish_phase()

    # =====================================================
    # ==================  VISION CALLBACKS  ===============
    # =====================================================
    def _on_vision_done(self, msg: Bool):
        self.vision_done = bool(msg.data)
        self.get_logger().info(f'Received soes_done = {self.vision_done}')

    def _on_centers(self, msg: CupcakeCenters):
        self.centers = [(p.x, p.y, p.z) for p in msg.centers]

        if len(self.centers) < 3:
            self.get_logger().warn("centers < 3; waiting for all three targets")
            return

        if self.active_index in (0, 1, 2) and self.arm_phase in (ArmPhase.HOME, ArmPhase.WAIT):
            self.get_logger().info(
                f"[ROBOHAND] Centers ready with active_index={self.active_index}, realigning phase."
            )
            self._align_arm_phase_with_index()

    # =====================================================
    # ==================  GENERIC HELPERS  ================
    # =====================================================
    def _enter(self, new_phase: Phase):
        old_phase = self.phase
        self.phase = new_phase
        self.phase_t0 = self.get_clock().now()
        self._did_start_pump = False

        # reset roller state jika bukan ROLL_TRAY
        if new_phase != Phase.ROLL_TRAY:
            self._roller_active = False
            self._roller_duration_s = 0.0

        # Vision request jika CAMERA
        if new_phase == Phase.CAMERA:
            self.vision_done = None
            req = Bool()
            req.data = True
            self.vision_request_pub.publish(req)
            self.get_logger().info('CAMERA: sent /vision/request = True, waiting /vision/soes_done')

        # Log overall
        self.get_logger().warn(f'[OVERALL] {old_phase.name} → {new_phase.name}')
        self.get_logger().info(f'[STATE] -> {self.phase.name}')
        self._publish_phase()

    def _elapsed(self) -> float:
        return (self.get_clock().now() - self.phase_t0).nanoseconds * 1e-9

    def _publish_index(self, idx: int):
        idx = int(idx)

        # NEW: leaving HOME (-1) clears home-hold latch
        if idx != -1:
            self._home_hold = False

        self.active_index = idx
        msg = Int32()
        msg.data = self.active_index
        self.index_pub.publish(msg)
        self.get_logger().info(f'active_index = {self.active_index}')
        self._align_arm_phase_with_index()

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

    # =====================================================
    # ==================  SWITCH / PAUSE  =================
    # =====================================================
    def _on_switch(self, msg: Bool):
        """
        /esp_switch_on:
        - True  = HIGH (tombol tidak ditekan)
        - False = LOW  (tombol ditekan)

        LOW (tekan)  -> mulai dari INIT_POS
        HIGH awal    -> tetap IDLE
        """
        prev = self.switch_on
        self.switch_on = bool(msg.data)

        # HIGH -> LOW : tombol ditekan -> RESET & mulai sequence
        if prev and not self.switch_on:
            self.get_logger().warn('RESET pressed (HIGH -> LOW) -> INIT_POS.')
            self.pump.stop()
            self._roller_cmd(False)
            self.swirl_active = False
            self._set_arm_at(False)
            self._step_idx = 0

            # HOME
            self._publish_index(-1)
            self._enter(Phase.INIT_POS)

        # LOW -> HIGH : tombol dilepas -> kembali ke IDLE
        elif (not prev) and self.switch_on:
            self.get_logger().warn('RESET released (LOW -> HIGH) -> IDLE.')
            self.pump.stop()
            self._roller_cmd(False)
            self._publish_index(-1)
            self._enter(Phase.IDLE)

    def _on_paused(self, msg: Bool):
        new_state = bool(msg.data)

        if new_state and not self.paused:
            self.pause_start = self.get_clock().now()

        elif not new_state and self.paused:
            if self.pause_start is not None:
                dt = self.get_clock().now() - self.pause_start
                # shift both high-level and arm timers
                self.phase_t0 = self.phase_t0 + dt
                self.arm_phase_t0 = self.arm_phase_t0 + dt
                self.pause_start = None

        self.paused = new_state

    def on_quality(self, msg: VisionQuality):
        self.quality_flag = bool(msg.needs_human)

    # =====================================================
    # ==================  MAIN TICK (STATE) ===============
    # =====================================================
    def tick(self):
        if self.paused:
            return

        if self.phase == Phase.TEST_MOTOR:
            self._test_motor_tick()
            return

        # INIT_POS: tunggu sampai arm HOME + settle
        if self.phase == Phase.INIT_POS:
            if self.arm_at and self.arm_at_since is not None:
                if (self.get_clock().now() - self.arm_at_since) >= Duration(seconds=self.t_settle):

                    if self._step_idx == 0:
                        self._start_step(0)

                    elif self._step_idx == 1:
                        self._start_step(1)

                    elif self._step_idx == 2:
                        self._start_step(2)

                    else:
                        self.get_logger().info("All steps done -> POST_STEP")
                        self._publish_index(-1)
                        self._enter(Phase.POST_STEP)

        elif self.phase == Phase.STEP0:
            if self._run_step():
                self.get_logger().info("STEP0 → STEP1")
                self._step_idx = 1
                self._publish_index(-1)
                self._enter(Phase.INIT_POS)

        elif self.phase == Phase.STEP1:
            if self._run_step():
                self.get_logger().info("STEP1 → STEP2")
                self._step_idx = 2
                self._publish_index(-1)
                self._enter(Phase.INIT_POS)

        elif self.phase == Phase.STEP2:
            if self._run_step():
                self.get_logger().info("STEP2 done -> INIT_POS before CAMERA")
                self._step_idx = 3
                self._publish_index(-1)
                self._enter(Phase.INIT_POS)

        # POST_STEP: cukup tunggu HOME settle lalu masuk CAMERA
        elif self.phase == Phase.POST_STEP:
            if self.arm_at and self.arm_at_since is not None:
                if (self.get_clock().now() - self.arm_at_since) >= Duration(seconds=self.t_settle):
                    self._enter(Phase.CAMERA)

        # CAMERA: tunggu vision_done atau timeout
        elif self.phase == Phase.CAMERA:

            # 1. Ada hasil
            if self.vision_done is not None:
                self.get_logger().info("Vision is done")
                if self.vision_done:
                    self.get_logger().info("Vision OK → ROLL_TRAY")
                else:
                    self.get_logger().warn("Vision BAD → ROLL_TRAY (tetap jalan tray)")
                self._enter(Phase.ROLL_TRAY)
                return

            # 2. Timeout
            if self._elapsed() >= self.cam_to:
                self.get_logger().warn("Camera timeout → ROLL_TRAY")
                self._enter(Phase.ROLL_TRAY)
                return

        # ROLL_TRAY
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

                self._publish_index(-1)
                self._step_idx = 0
                self._enter(Phase.INIT_POS)

        elif self.phase == Phase.IDLE:
            return

    # =====================================================
    # ==================  STEP LOGIC  =====================
    # =====================================================
    def _start_step(self, step_idx: int):
        idx = self.order[step_idx]
        self._publish_index(idx)
        self._enter(
            Phase.STEP0 if step_idx == 0
            else Phase.STEP1 if step_idx == 1
            else Phase.STEP2
        )

    def _run_step(self) -> bool:
        """
        True  -> step selesai (swirl berhenti, pump dimatikan)
        False -> masih swirl / masih jalan
        """
        if self.swirl_active:
            if not self._did_start_pump:
                self.pump.start(duty=1.0, duration_s=0.0)
                self._did_start_pump = True
                self.get_logger().info('Pump ON (SWIRL)')
            return False
        else:
            if self._did_start_pump:
                self.pump.stop()
                self._did_start_pump = False
                self.get_logger().info('Pump OFF (step complete)')
                return True
            return False

    # =====================================================
    # ==================  TEST_MOTOR   ====================
    # =====================================================
    def _test_motor_tick(self):
        t = self._elapsed()
        period = self.test_period_s
        segment = int(t // period)
        direction = 1.0 if (segment % 2) == 0 else -1.0

        jt = JointTargets()
        jt.position = [0.0, 0.0, 0.0, 0.0]
        jt.velocity = [0.0, 0.0, 0.0, 0.0]
        jt.use_velocity = False

        servo_neutral_deg = 90.0
        servo_low_deg, servo_high_deg = self.test_servo_deg
        jt.position[3] = math.radians(servo_neutral_deg)

        pump_msg = PumpCmd()
        pump_msg.on = False
        pump_msg.duty = 0.0
        pump_msg.duration_s = 0.0

        amp0, amp1, amp2 = self.test_amp_rad

        if segment == 0:
            jt.position[0] = direction * amp0

        elif segment == 1:
            jt.position[1] = direction * amp1

        elif segment == 2:
            jt.position[2] = direction * amp2

        elif segment == 3:
            angle_deg = servo_high_deg if direction > 0 else servo_low_deg
            jt.position[3] = math.radians(angle_deg)

        elif segment == 4:
            jt.position[0] = direction * amp0
            jt.position[1] = direction * amp1
            jt.position[2] = direction * amp2
            angle_deg = servo_high_deg if direction > 0 else servo_low_deg
            jt.position[3] = math.radians(angle_deg)

        else:
            jt.position = [0.0, 0.0, 0.0, math.radians(servo_neutral_deg)]
            pump_msg.on = True
            pump_msg.duty = 1.0
            pump_msg.duration_s = 0.0

        # Saat TEST_MOTOR, IK loop di-_arm_tick akan dilewati (lihat _arm_tick)
        self.pump_pub.publish(pump_msg)
        self.arm_pub.publish(jt)

    # =====================================================
    # ==================  ARM / IK PART  ==================
    # =====================================================
    def _set_arm_at(self, is_at: bool):
        if is_at:
            if not self.arm_at:
                self.arm_at_since = self.get_clock().now()
            self.arm_at = True
        else:
            self.arm_at = False
            self.arm_at_since = None

        self.arm_at_pub.publish(Bool(data=self.arm_at))

    def _set_swirl_active(self, active: bool):
        self.swirl_active = bool(active)
        self.swirl_pub.publish(Bool(data=self.swirl_active))

    def _align_arm_phase_with_index(self):
        # active_index == -1 -> HOME or HOLD (latched)
        if self.active_index == -1:
            if self._home_hold and self.arm_phase == ArmPhase.WAIT:
                # keep holding, do not re-run HOME IK
                self._publish_targets(self.q, np.zeros(4, dtype=float), use_velocity=True)
                self._set_arm_at(True)
                self._set_swirl_active(False)
                return

            self.get_logger().info("[ROBOHAND] Moving to init pos (HOME)")
            self._arm_enter(ArmPhase.HOME, None)
            return

        # 0/1/2 -> MOVE ke salah satu cupcake center
        if self.active_index in (0, 1, 2) and self.centers and len(self.centers) >= 3:
            label = f"pos{self.active_index + 1}"
            self.get_logger().info(f"[ROBOHAND] Moving to {label}")
            self._arm_enter(ArmPhase.MOVE, np.array(self.centers[self.active_index], dtype=float))
            return

        self._arm_enter(ArmPhase.WAIT, None)

    def _arm_enter(self, new_phase: ArmPhase, xyz: Optional[np.ndarray]):
        old_phase = self.arm_phase

        self.arm_phase = new_phase
        self.arm_phase_t0 = self.get_clock().now()
        self.last_within_tol = None
        self.des_xyz = xyz.copy() if xyz is not None else None

        # reset braking whenever phase changes
        self._braking = False
        self._brake_t0 = None
        self._brake_qdot0 = np.zeros(4, dtype=float)

        # reset stability gate
        self._stable_count = 0

        # reset HOME arrival log
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

    def _s_curve_speed(self, profile_T: float) -> float:
        """
        Smooth S-curve-like speed factor in [0.1, 1].
        """
        if profile_T <= 0.0:
            return 1.0
        t = self._arm_elapsed()
        tau = max(0.0, min(t / profile_T, 1.0))
        scale = 4.0 * tau * (1.0 - tau)
        return max(scale, 0.1)

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

    def _home_step(self, speed_scale: float = 1.0) -> bool:
        """Home motion di task space pakai IK yang sama."""
        des_xyz_home = self.fk_xyz(self.q_home)
        at = self._ik_step(des_xyz_home, xdot_ff=None, speed_scale=speed_scale)

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
        """Cartesian IK step dengan S-curve speed scaling."""
        cur_xyz = self.fk_xyz(self.q)
        err = des_xyz - cur_xyz
        err_norm = float(np.linalg.norm(err))

        v = self.kp * err
        if xdot_ff is not None:
            v = v + xdot_ff

        J = self.jacobian(self.q)
        JJt = J @ J.T
        qdot = J.T @ np.linalg.solve(JJt + (self.lmbda**2) * np.eye(3), v)

        # S-curve speed scaling
        limit = self.qdot_lim * speed_scale
        qdot = np.clip(qdot, -limit, limit)

        vel_norm = float(np.max(np.abs(qdot)))

        # -------- Stability gate (anti-cetek) --------
        pos_ok = err_norm <= self.pos_tol
        vel_ok = vel_norm <= self.vel_eps

        if pos_ok and vel_ok:
            self._stable_count += 1
        else:
            self._stable_count = 0
            self._braking = False
            self._brake_t0 = None

        at = (self._stable_count >= max(1, self.stable_cycles))

        # -------- Soft braking ONLY when stable --------
        if at:
            now = self.get_clock().now()

            if not self._braking:
                self._braking = True
                self._brake_t0 = now
                self._brake_qdot0 = qdot.copy()

            T = max(self.brake_T, 1e-3)
            t = (now - self._brake_t0).nanoseconds * 1e-9
            k = max(0.0, 1.0 - (t / T))  # linear ramp down

            qdot_cmd = self._brake_qdot0 * k

            # keep updating internal q during braking
            self.q = np.clip(self.q + qdot_cmd * self.dt, self.q_min, self.q_max)

            # keep velocity mode to avoid "cetek" on mode switch
            self._publish_targets(self.q, qdot_cmd, use_velocity=True)

            if k <= 0.0:
                self._set_arm_at(True)
                return True
            else:
                self._set_arm_at(False)
                return False

        # -------- Normal IK when not "at" --------
        self.q = np.clip(self.q + qdot * self.dt, self.q_min, self.q_max)
        self._publish_targets(self.q, qdot, use_velocity=True)

        self._set_arm_at(False)
        return False

    def _start_swirl(self):
        # siapkan spiral di sekitar center aktif
        if self.centers is None or self.active_index not in (0, 1, 2):
            return

        label = f"pos{self.active_index + 1}"
        self.get_logger().info(f"[ROBOHAND] Arrived at {label}, starting swirl")

        self.spiral_center = np.array(self.centers[self.active_index], dtype=float)
        self.spiral_theta = 0.0
        self._arm_enter(ArmPhase.SWIRL, self.spiral_center.copy())

    def _arm_tick(self):
        if self.paused:
            return

        # Saat TEST_MOTOR, jangan kirim IK
        if self.phase == Phase.TEST_MOTOR:
            return

        # HOME
        if self.arm_phase == ArmPhase.HOME:
            speed_scale = self._s_curve_speed(self.home_T)
            done = self._home_step(speed_scale=speed_scale)

            # NEW: latch HOLD when HOME is reached, stop re-sending tiny velocities later
            if done:
                self._home_hold = True
                self._publish_targets(self.q, np.zeros(4, dtype=float), use_velocity=True)
                self._set_swirl_active(False)
                self._set_arm_at(True)
                self._arm_enter(ArmPhase.WAIT, None)
                return

            self._set_swirl_active(False)
            return

        # WAIT
        if self.arm_phase == ArmPhase.WAIT:
            # if holding HOME, keep sending 0 vel (optional but makes ESP steady)
            if self.active_index == -1 and self._home_hold:
                self._publish_targets(self.q, np.zeros(4, dtype=float), use_velocity=True)
                self._set_arm_at(True)
            else:
                self._set_arm_at(False)

            self._set_swirl_active(False)
            return

        # MOVE
        if self.arm_phase == ArmPhase.MOVE and self.des_xyz is not None:
            speed_scale = self._s_curve_speed(self.move_T)
            at = self._ik_step(self.des_xyz, speed_scale=speed_scale)
            if at:
                self._start_swirl()
            self._set_swirl_active(False)
            return

        # SWIRL
        if self.arm_phase == ArmPhase.SWIRL and self.spiral_center is not None:
            if self.theta_max <= 0.0:
                self._arm_enter(ArmPhase.WAIT, None)
                self._set_swirl_active(False)
                return

            # Spiral pose
            r = self.R0 * (1.0 + self.alpha * self.spiral_theta)
            dx = r * math.cos(self.spiral_theta)
            dy = r * math.sin(self.spiral_theta)
            dz = self.s * self.spiral_theta
            des = self.spiral_center + np.array([dx, dy, dz])

            # Spiral feedforward
            rdot = self.R0 * self.alpha * self.omega
            xdot = rdot * math.cos(self.spiral_theta) - r * self.omega * math.sin(self.spiral_theta)
            ydot = rdot * math.sin(self.spiral_theta) + r * self.omega * math.cos(self.spiral_theta)
            zdot = self.s * self.omega
            ff = np.array([xdot, ydot, zdot])

            self._ik_step(des, ff, speed_scale=1.0)
            self.spiral_theta += self.omega * self.dt

            if self.spiral_theta >= self.theta_max:
                label = f"pos{self.active_index + 1}" if self.active_index in (0, 1, 2) else "current position"
                self.get_logger().info(f"[SWIRL] Swirl done at {label}")
                self._arm_enter(ArmPhase.WAIT, None)
                self._set_swirl_active(False)
            else:
                self.get_logger().debug("[SWIRL] Active")
                self._set_swirl_active(True)
            return

        # default
        self._set_arm_at(False)
        self._set_swirl_active(False)


def main():
    rclpy.init()
    node = StateNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
