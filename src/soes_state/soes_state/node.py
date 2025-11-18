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
    STEP        = 1        # <---- NEW unified step
    CAMERA      = 2
    ROLL_TRAY   = 3
    IDLE        = 4
    TEST_MOTOR  = 5


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

        # TEST_MOTOR params
        self.declare_parameter('test_period_s', 3.0)
        self.declare_parameter('test_amp_rad', [0.4, 0.4, 0.4])
        self.declare_parameter('test_servo_deg', [30.0, 150.0])
        self.test_period_s = float(self.get_parameter('test_period_s').value)
        self.test_amp_rad  = list(self.get_parameter('test_amp_rad').value)
        self.test_servo_deg = list(self.get_parameter('test_servo_deg').value)

        # ---------- ROS I/O ----------
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.index_pub = self.create_publisher(Int32, '/state/active_index', 1)
        self.pump_pub  = self.create_publisher(PumpCmd, '/pump/cmd', 1)
        self.roll_cli  = self.create_client(RollTray, '/tray/roll')
        self.qual_sub  = self.create_subscription(VisionQuality, '/vision/quality', self.on_quality, qos)
        self.arm_pub   = self.create_publisher(JointTargets, '/arm/joint_targets', 10)

        # NEW: subscribe to /arm/at_target to GATE transitions
        self.arm_at       = False
        self.arm_at_since = None
        self.create_subscription(Bool, '/arm/at_target', self._on_at_target, 10)

        # NEW: subscribe to /esp_switch_on to gate the whole state machine
        self.switch_on = False
        self.create_subscription(Bool, '/esp_switch_on', self._on_switch, 10)

        # NEW: subscribe to /arm/swirl_active to control pump
        self.swirl_active = False
        self.create_subscription(Bool, '/arm/swirl_active', self._on_swirl, 10)

        # NEW: subscribe to /esp_paused to globally pause
        self.paused = False
        self.pause_start = None
        self.create_subscription(Bool, '/esp_paused', self._on_paused, 10)

        # Pump helper
        self.pump = PumpController(self._pump_on, self._pump_off)

        # ---------- Runtime ----------
        self.phase = Phase.INIT_POS
        self.phase_t0 = self.get_clock().now()
        self.quality_flag = False
        self._step_idx = 0    # 0 → 1 → 2 for three spirals
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
        self.switch_on = bool(msg.data)

    def _on_swirl(self, msg: Bool):
        self.swirl_active = bool(msg.data)

    def _on_paused(self, msg: Bool):
        new_state = bool(msg.data)
        if new_state and not self.paused:
            self.pause_start = self.get_clock().now()
        elif not new_state and self.paused:
            if self.pause_start is not None:
                dt = self.get_clock().now() - self.pause_start
                self.phase_t0 = self.phase_t0 + dt
                self.pause_start = None
        self.paused = new_state

    # ---------- Callbacks ----------
    def on_quality(self, msg: VisionQuality):
        self.quality_flag = bool(msg.needs_human)

    # ---------- Main tick ----------
    def tick(self):
        if self.paused:
            return

        if self.phase == Phase.TEST_MOTOR:
            self._test_motor_tick()
            return

        # SWITCH OFF => IDLE
        if not self.switch_on:
            if self.phase != Phase.IDLE:
                self.get_logger().warn('ESP switch OFF → entering IDLE.')
                self.pump.stop()
                self._publish_index(-1)
                self._enter(Phase.IDLE)
            return

        # SWITCH turns ON while IDLE: restart
        if self.phase == Phase.IDLE:
            self.get_logger().info('ESP switch ON → restarting INIT_POS.')
            self._publish_index(-1)
            self._enter(Phase.INIT_POS)
            return

        # ======== NORMAL SEQUENCE ========
        if self.phase == Phase.INIT_POS:
            # Wait for HOME settle
            if self.arm_at and self.arm_at_since is not None:
                if (self.get_clock().now() - self.arm_at_since) >= Duration(seconds=self.t_settle):

                    # Run 3 spirals: 0,1,2
                    if self._step_idx < 3:
                        self._start_step(self._step_idx)
                    else:
                        self._enter(Phase.CAMERA)

        elif self.phase == Phase.STEP:
            if self._run_step():
                self._step_idx += 1
                self._publish_index(-1)
                self._enter(Phase.INIT_POS)

        elif self.phase == Phase.CAMERA:
            if self._elapsed() >= self.cam_to:
                if self.quality_flag:
                    self.get_logger().warn('quality check requests attention.')
                else:
                    self.get_logger().info('quality check OK.')
                self._enter(Phase.ROLL_TRAY)

        elif self.phase == Phase.ROLL_TRAY:
            if not self.roll_cli.service_is_ready():
                self.get_logger().info('Waiting for /tray/roll ...')
                return

            req = RollTray.Request()
            req.distance_mm = self.roll_dist
            req.speed_mm_s  = self.roll_speed
            self.roll_cli.call_async(req)

            # Restart full sequence
            self._publish_index(-1)
            self._step_idx = 0
            self._enter(Phase.INIT_POS)

        elif self.phase == Phase.IDLE:
            return

    # ---------- Step logic ----------
    def _start_step(self, step_idx: int):
        idx = self.order[step_idx]
        self._publish_index(idx)
        self._enter(Phase.STEP)

    def _run_step(self):
        """
        Pump follows RoboHand SWIRL:
          - swirl_active True  -> pump ON
          - swirl_active False after True -> pump OFF and complete step
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

        self.pump_pub.publish(pump_msg)
        self.arm_pub.publish(jt)


def main():
    rclpy.init()
    node = StateNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
