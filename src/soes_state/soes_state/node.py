#!/usr/bin/env python3
import enum, math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration

from std_msgs.msg import Bool, Int32
from soes_msgs.msg import PumpCmd, VisionQuality, JointTargets
from soes_msgs.srv import RollTray

from .utils import PumpController


class Phase(enum.Enum):
    INIT_POS    = 0
    STEP0       = 1
    # STEP1 and later phases are preserved but UNUSED in the "single-step" mode requested.
    STEP1       = 2  # UNUSED in single-step mode
    STEP2       = 3  # UNUSED in single-step mode
    CAMERA      = 4  # UNUSED in single-step mode
    ROLL_TRAY   = 5  # UNUSED in single-step mode
    IDLE        = 6
    TEST_MOTOR  = 7  # retained for manual testing, UNUSED in normal single-step mode


class StateNode(Node):
    def __init__(self):
        super().__init__('soes_state')

        # ---------- Parameters ----------
        self.declare_parameter('settle_before_pump_s', 0.6)
        self.declare_parameter('pump_on_s', 2.0)
        self.declare_parameter('swirl_time_s', 1.0)
        self.declare_parameter('order', [0, 1, 2])

        self.declare_parameter('roller_distance_mm', 100.0)
        self.declare_parameter('roller_speed_mm_s', 40.0)

        self.declare_parameter('camera_timeout_s', 2.0)

        # also use settle_before_pump_s as "arm_home_settle_s"
        self.t_settle   = float(self.get_parameter('settle_before_pump_s').value)
        self.t_pump     = float(self.get_parameter('pump_on_s').value)
        self.t_swirl    = float(self.get_parameter('swirl_time_s').value)
        self.order      = list(self.get_parameter('order').value)
        self.roll_dist  = float(self.get_parameter('roller_distance_mm').value)
        self.roll_speed = float(self.get_parameter('roller_speed_mm_s').value)
        self.cam_to     = float(self.get_parameter('camera_timeout_s').value)

        # TEST_MOTOR params (kept but UNUSED in single-step normal mode)
        self.declare_parameter('test_period_s', 3.0)         # UNUSED in single-step
        self.declare_parameter('test_amp_rad', [0.4, 0.4, 0.4])  # UNUSED in single-step
        self.declare_parameter('test_servo_deg', [30.0, 150.0])  # UNUSED in single-step
        self.test_period_s = float(self.get_parameter('test_period_s').value)
        self.test_amp_rad  = list(self.get_parameter('test_amp_rad').value)
        self.test_servo_deg = list(self.get_parameter('test_servo_deg').value)

        # ---------- ROS I/O ----------
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.index_pub = self.create_publisher(Int32, '/state/active_index', 1)
        self.pump_pub  = self.create_publisher(PumpCmd, '/pump/cmd', 1)
        self.roll_cli  = self.create_client(RollTray, '/tray/roll')  # UNUSED in single-step mode (roll not invoked)
        self.qual_sub  = self.create_subscription(VisionQuality, '/vision/quality', self.on_quality, qos)  # UNUSED in single-step mode
        self.arm_pub   = self.create_publisher(JointTargets, '/arm/joint_targets', 10)  # only used in TEST_MOTOR / manual tests

        # NEW: subscribe to /arm/at_target to GATE transitions
        self.arm_at       = False
        self.arm_at_since = None
        self.create_subscription(Bool, '/arm/at_target', self._on_at_target, 10)

        # NEW: subscribe to /esp_switch_on to gate the whole state machine
        self.switch_on = False  # default OFF -> goes to IDLE until switch turns ON
        self.create_subscription(Bool, '/esp_switch_on', self._on_switch, 10)

        # NEW: subscribe to /arm/swirl_active to control pump
        self.swirl_active = False
        self.create_subscription(Bool, '/arm/swirl_active', self._on_swirl, 10)

        # NEW: subscribe to /esp_paused to globally pause state machine
        self.paused = False
        self.pause_start = None
        self.create_subscription(Bool, '/esp_paused', self._on_paused, 10)

        # Pump helper
        self.pump = PumpController(self._pump_on, self._pump_off)

        # ---------- Runtime ----------
        # Start at INIT_POS. The behavior is now: INIT_POS -> STEP0 -> IDLE (done).
        self.phase = Phase.INIT_POS      # set TEST_MOTOR or INIT_POS
        self.phase_t0 = self.get_clock().now()
        self.quality_flag = False
        self._step_idx = 0
        self._did_start_pump = False

        # 20 Hz tick
        self.timer = self.create_timer(0.05, self.tick)
        self.get_logger().info('soes_state: ready (INIT_POS).')

        # Tell robothand to go HOME
        self._publish_index(-1)

    # ---------- Helpers ----------
    def _enter(self, new_phase: Phase):
        self.phase = new_phase
        self.phase_t0 = self.get_clock().now()
        self._did_start_pump = False
        self.get_logger().info(f'[STATE] -> {self.phase.name}')

    def _elapsed(self) -> float:
        return (self.get_clock().now() - self.phase_t0).nanoseconds * 1e-9

    def _publish_index(self, idx: int):
        msg = Int32(); msg.data = int(idx)
        self.index_pub.publish(msg)
        self.get_logger().info(f'active_index = {idx}')

    def _pump_on(self, duty: float, duration_s: float):
        msg = PumpCmd(); msg.on = True; msg.duty = float(duty); msg.duration_s = float(duration_s)
        self.pump_pub.publish(msg)

    def _pump_off(self):
        msg = PumpCmd(); msg.on = False; msg.duty = 0.0; msg.duration_s = 0.0
        self.pump_pub.publish(msg)

    def _on_at_target(self, msg: Bool):
        if msg.data:
            if not self.arm_at:
                self.arm_at_since = self.get_clock().now()
            self.arm_at = True
        else:
            self.arm_at = False
            self.arm_at_since = None

    def _on_switch(self, msg: Bool):
        """
        /esp_switch_on:
          - False -> force state machine into IDLE
          - True  -> if currently IDLE, restart from INIT_POS
        """
        self.switch_on = bool(msg.data)

    def _on_swirl(self, msg: Bool):
        """Track when RoboHand is in SWIRL phase."""
        self.swirl_active = bool(msg.data)

    def _on_paused(self, msg: Bool):
        """
        /esp_paused:
          - True  -> freeze state machine (tick does nothing).
          - False -> resume, adjusting timers so elapsed() ignores pause duration.
        """
        new_state = bool(msg.data)

        # Entering pause
        if new_state and not self.paused:
            self.pause_start = self.get_clock().now()

        # Leaving pause
        elif not new_state and self.paused:
            if self.pause_start is not None:
                dt = self.get_clock().now() - self.pause_start
                # Shift phase_t0 forward by pause duration so _elapsed() doesn't count it
                self.phase_t0 = self.phase_t0 + dt
                self.pause_start = None

        self.paused = new_state

    # ---------- Callbacks ----------
    def on_quality(self, msg: VisionQuality):
        # This quality check path is preserved but UNUSED in the single-step flow.
        self.quality_flag = bool(msg.needs_human)

    # ---------- Main tick ----------
    def tick(self):
        # ======== GLOBAL PAUSE GATING ========
        if self.paused:
            # Do nothing while ESP pause button is active:
            # - phase is kept
            # - timers are adjusted on resume (see _on_paused)
            return

        # ======== TEST_MOTOR ========
        if self.phase == Phase.TEST_MOTOR:
            # TEST_MOTOR retained for hardware checks; not part of the single-step normal flow.
            self._test_motor_tick()
            return

        # ======== SWITCH GATING (ESP) ========
        if not self.switch_on:
            # Switch OFF → go/stay in IDLE, ensure pump OFF and arm commanded HOME
            if self.phase != Phase.IDLE:
                self.get_logger().warn('ESP switch OFF → entering IDLE.')
                self.pump.stop()
                self._publish_index(-1)   # go to HOME
                self._enter(Phase.IDLE)
            return
        else:
            # Switch turned ON while in IDLE → restart from INIT_POS
            if self.phase == Phase.IDLE:
                self.get_logger().info('ESP switch ON → restarting from INIT_POS.')
                self._publish_index(-1)   # command HOME again
                self._enter(Phase.INIT_POS)
                # fall through into normal INIT_POS handling below

        # ======== NORMAL SEQUENCE ========
        if self.phase == Phase.INIT_POS:
            # WAIT for robothand to confirm it is at HOME (via /arm/at_target)
            if self.arm_at and self.arm_at_since is not None:
                if (self.get_clock().now() - self.arm_at_since) >= Duration(seconds=self.t_settle):

                    # NEW SEQUENCE LOGIC: Only start STEP0 (single-step mode)
                    if self._step_idx == 0:
                        self._start_step(0)

                    # Note: _step_idx is never advanced beyond 0 in this single-step mode,
                    # so STEP1/STEP2/CAMERA/ROLL_TRAY branches are effectively unused.
                    # They remain in code for backward compatibility / future re-enable.

        elif self.phase == Phase.STEP0:
            # Run STEP0. When it completes, stop the pump, publish HOME and go to IDLE (done).
            if self._run_step():
                self.get_logger().info("STEP0 complete → DONE (entering IDLE).")
                # Ensure pump is off
                self.pump.stop()
                # Command HOME
                self._publish_index(-1)
                # Enter IDLE to stop the state machine (single-step complete)
                self._enter(Phase.IDLE)

        elif self.phase == Phase.STEP1:
            # UNUSED in single-step mode: preserved for future multi-step operation
            if self._run_step():
                self.get_logger().info("STEP1 complete → STEP2")
                self._step_idx = 2
                self._publish_index(-1)
                self._enter(Phase.INIT_POS)

        elif self.phase == Phase.STEP2:
            # UNUSED in single-step mode: preserved for future multi-step operation
            if self._run_step():
                self.get_logger().info("STEP2 complete → CAMERA")
                self._step_idx = 3  # may or may not be used later
                self._publish_index(-1)
                self._enter(Phase.INIT_POS)

        elif self.phase == Phase.CAMERA:
            # UNUSED in single-step mode
            if self._elapsed() >= self.cam_to:
                if self.quality_flag:
                    self.get_logger().warn('quality check requests attention.')
                else:
                    self.get_logger().info('quality check OK.')
                self._enter(Phase.ROLL_TRAY)

        elif self.phase == Phase.ROLL_TRAY:
            # UNUSED in single-step mode
            if not self.roll_cli.service_is_ready():
                self.get_logger().info('Waiting for /tray/roll ...')
                return

            req = RollTray.Request()
            req.distance_mm = self.roll_dist
            req.speed_mm_s  = self.roll_speed
            self.roll_cli.call_async(req)

            # restart the cycle (unused for single-step)
            self._publish_index(-1)     # back to HOME
            self._step_idx = 0          # reset sequence
            self._enter(Phase.INIT_POS)

        elif self.phase == Phase.IDLE:
            # IDLE is the final state in single-step mode
            return

    # ---------- Step logic ----------
    def _start_step(self, step_idx: int):
        idx = self.order[step_idx]
        self._publish_index(idx)
        # Only STEP0 is used in this single-step flow; phases for STEP1/STEP2 retained but unused.
        self._enter(Phase.STEP0 if step_idx == 0 else Phase.STEP1 if step_idx == 1 else Phase.STEP2)

    def _run_step(self):
        """
        Pump now follows RoboHand SWIRL phase:
          - while /arm/swirl_active == True  -> pump ON
          - when it returns to False after having been True -> pump OFF and step complete

        This behavior is unchanged and is used for STEP0 in the single-step flow.
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
                self.get_logger().info('Pump OFF (SWIRL complete)')
                return True
            return False

    # ---------- TEST_MOTOR helper ----------
    def _test_motor_tick(self):
        """
        Simple hardware test:
          seg0: J0 swings ±test_amp_rad[0]
          seg1: J1 swings ±test_amp_rad[1]
          seg2: J2 swings ±test_amp_rad[2]
          seg3: servo toggles between low/high degrees
          seg4: all joints move together
          seg5+: pump ON (constant), joints neutral

        NOTE: TEST_MOTOR is retained but not part of the normal single-step flow.
        """
        t = self._elapsed()
        period = self.test_period_s

        # segment index increases every `period` seconds
        segment = int(t // period)

        # direction flips every full period: +1, -1, +1, -1, ...
        direction = 1.0 if (segment % 2) == 0 else -1.0

        jt = JointTargets()
        jt.position = [0.0, 0.0, 0.0, 0.0]
        jt.velocity = [0.0, 0.0, 0.0, 0.0]
        jt.use_velocity = False  # we stay in position mode for tests

        # Servo neutral / range from parameters
        servo_neutral_deg = 90.0
        servo_low_deg, servo_high_deg = self.test_servo_deg
        jt.position[3] = math.radians(servo_neutral_deg)

        # Base pump command (OFF by default)
        pump_msg = PumpCmd()
        pump_msg.on = False
        pump_msg.duty = 0.0
        pump_msg.duration_s = 0.0

        # Use configured amplitudes from parameters
        amp0, amp1, amp2 = self.test_amp_rad

        if segment == 0:
            # Test J0 only
            jt.position[0] = direction * amp0

        elif segment == 1:
            # Test J1 only
            jt.position[1] = direction * amp1

        elif segment == 2:
            # Test J2 only
            jt.position[2] = direction * amp2

        elif segment == 3:
            # Test servo: flip between low/high angles
            angle_deg = servo_high_deg if direction > 0 else servo_low_deg
            jt.position[3] = math.radians(angle_deg)

        elif segment == 4:
            # All joints move together
            jt.position[0] = direction * amp0
            jt.position[1] = direction * amp1
            jt.position[2] = direction * amp2
            angle_deg = servo_high_deg if direction > 0 else servo_low_deg
            jt.position[3] = math.radians(angle_deg)

        else:
            # Pump test only: joints neutral, pump ON at full duty
            jt.position = [0.0, 0.0, 0.0, math.radians(servo_neutral_deg)]
            pump_msg.on = True
            pump_msg.duty = 1.0
            pump_msg.duration_s = 0.0  # continuous until we leave TEST_MOTOR

        # Publish commands
        self.pump_pub.publish(pump_msg)
        self.arm_pub.publish(jt)


def main():
    rclpy.init()
    node = StateNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
