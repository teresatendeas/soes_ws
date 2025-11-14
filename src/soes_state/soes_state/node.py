import enum
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Int32
from soes_msgs.msg import PumpCmd, VisionQuality
from soes_msgs.srv import RollTray
from soes_msgs.msg import JointTargets
import math

from .utils import PumpController


class Phase(enum.Enum):
    INIT_POS    = 0
    STEP0       = 1   # index = order[0]
    STEP1       = 2   # index = order[1]
    STEP2       = 3   # index = order[2]
    CAMERA      = 4
    ROLL_TRAY   = 5
    IDLE        = 6
    TEST_MOTOR  = 7


class StateNode(Node):
    def __init__(self):
        super().__init__('soes_state')

        # ---------- Parameters ----------
        # Timing (state only; robothand handles MOVE/SWIRL internally)
        self.declare_parameter('settle_before_pump_s', 0.6)  # wait after setting index before pump
        self.declare_parameter('pump_on_s', 2.0)             # pump ON duration
        self.declare_parameter('swirl_time_s', 1.0)          # time to let robothand lift (+3 cm)
        self.declare_parameter('order', [0, 1, 2])           # visit order for centers

        # Roller
        self.declare_parameter('roller_distance_mm', 100.0)
        self.declare_parameter('roller_speed_mm_s', 40.0)

        # Vision
        self.declare_parameter('camera_timeout_s', 2.0)

        self.t_settle   = float(self.get_parameter('settle_before_pump_s').value)
        self.t_pump     = float(self.get_parameter('pump_on_s').value)
        self.t_swirl    = float(self.get_parameter('swirl_time_s').value)
        self.order      = list(self.get_parameter('order').value)
        self.roll_dist  = float(self.get_parameter('roller_distance_mm').value)
        self.roll_speed = float(self.get_parameter('roller_speed_mm_s').value)
        self.cam_to     = float(self.get_parameter('camera_timeout_s').value)

        # Testing motors    
        self.declare_parameter('test_period_s', 3.0)      # flip direction every N seconds
        self.declare_parameter('test_amp_rad', [0.4, 0.4, 0.4])   # amplitudes for 3 steppers [rad]
        self.declare_parameter('test_servo_deg', [30.0, 150.0])   # servo low/high [deg]

        self.test_period_s = float(self.get_parameter('test_period_s').value)
        self.test_amp_rad  = list(self.get_parameter('test_amp_rad').value)
        self.test_servo_deg = list(self.get_parameter('test_servo_deg').value)
        self.arm_pub = self.create_publisher(JointTargets, '/arm/joint_targets', 10)


        qos = QoSProfile(depth=10,
                         reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)

        # ---------- ROS I/O ----------
        self.index_pub = self.create_publisher(Int32, '/state/active_index', 1)
        self.pump_pub  = self.create_publisher(PumpCmd, '/pump/cmd', 1)
        self.qual_sub  = self.create_subscription(VisionQuality, '/vision/quality', self.on_quality, qos)
        self.roll_cli  = self.create_client(RollTray, '/tray/roll')

        # Pump helper
        self.pump = PumpController(self._pump_on, self._pump_off)

        # ---------- Runtime ----------
        #self.phase = Phase.TEST_MOTOR # <-------------------------------------------------------- CHANGE THIS
        self.phase = Phase.INIT_POS
        self.phase_t0 = self.get_clock().now()
        self.quality_flag = False
        self._step_idx = 0  # 0..2 for STEP0..STEP2
        self._did_start_pump = False

        # 20 Hz tick
        self.timer = self.create_timer(0.05, self.tick)
        self.get_logger().info('soes_state: ready (INIT_POS).')

        # go to init pose (robothand interprets -1 as home)
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
        # Clamp if you want safety limits here
        msg = PumpCmd(); msg.on = True; msg.duty = float(duty); msg.duration_s = float(duration_s)
        self.pump_pub.publish(msg)

    def _pump_off(self):
        msg = PumpCmd(); msg.on = False; msg.duty = 0.0; msg.duration_s = 0.0
        self.pump_pub.publish(msg)

    # ---------- Callbacks ----------
    def on_quality(self, msg: VisionQuality):
        self.quality_flag = bool(msg.needs_human)

    # ---------- Main tick ----------
    def tick(self):
        # ======== TEST_MOTOR MODE ========
        if self.phase == Phase.TEST_MOTOR:
            t = self._elapsed()
            period = self.test_period_s

            # Which part of the test sequence we are in
            # 0: J0 one-by-one
            # 1: J1 one-by-one
            # 2: J2 one-by-one
            # 3: Servo 180° front/back
            # 4: All joints together
            # >=5: Pump swirl (+5 rev, pause, -5 rev, pause)
            segment = int(t // period)

            # Base joint command
            jt = JointTargets()
            jt.position = [0.0, 0.0, 0.0, 0.0]   # q0, q1, q2, servo
            jt.velocity = [0.0, 0.0, 0.0, 0.0]   # v0, v1, v2, pump (via vel[3])
            jt.use_velocity = False              # default = position mode

            # Servo neutral (for phases where it shouldn't move)
            servo_neutral_deg = 90.0
            jt.position[3] = math.radians(servo_neutral_deg)

            # PumpCmd default OFF (we'll drive pump via vel[3] in pump phase)
            pump_msg = PumpCmd()
            pump_msg.on = False
            pump_msg.duty = 0.0
            pump_msg.duration_s = 0.0

            # Flip direction every 'period' seconds: +1, -1, +1, ...
            direction = 1.0 if int(t / period) % 2 == 0 else -1.0

            # 360° for arm steppers, 180° for servo
            amp_rad = math.radians(360.0)  # 2π rad
            servo_low_deg, servo_high_deg = 0.0, 180.0

            mode = ""

            # --------- ARM + SERVO SEQUENCE ---------
            if segment == 0:
                # Joint 0 only: ±360°
                jt.position[0] = direction * amp_rad
                mode = "J0 one-by-one 360°"
            elif segment == 1:
                # Joint 1 only: ±360°
                jt.position[1] = direction * amp_rad
                mode = "J1 one-by-one 360°"
            elif segment == 2:
                # Joint 2 only: ±360°
                jt.position[2] = direction * amp_rad
                mode = "J2 one-by-one 360°"
            elif segment == 3:
                # Servo only: 0° ↔ 180°
                servo_deg = servo_high_deg if direction > 0 else servo_low_deg
                jt.position[3] = math.radians(servo_deg)
                mode = "Servo 180° front/back"
            elif segment == 4:
                # All 3 arm motors: ±360°, servo 0°/180° together
                jt.position[0] = direction * amp_rad
                jt.position[1] = direction * amp_rad
                jt.position[2] = direction * amp_rad
                servo_deg = servo_high_deg if direction > 0 else servo_low_deg
                jt.position[3] = math.radians(servo_deg)
                mode = "All 3 steppers 360° + servo 180°"
            else:
                # --------- PUMP SWIRL PHASE (J3 via velocity) ---------
                # ESP side:
                #   pos[3] = servo angle
                #   vel[3] = pump stepper velocity
                jt.use_velocity = True

                # Arms stay at rest
                jt.position[0] = 0.0
                jt.position[1] = 0.0
                jt.position[2] = 0.0

                # Servo stays neutral
                jt.position[3] = math.radians(servo_neutral_deg)

                # Time since the pump phase started
                t_pump = t - 5 * period

                pump_speed = 10.0  # rad/s; adjust if you want slower/faster swirl
                rev = 5.0
                # Time needed for 5 revolutions at pump_speed
                swirl_time = (rev * 2.0 * math.pi) / pump_speed

                # Pattern:
                #   0          .. swirl_time        -> +5 rev
                #   swirl_time .. swirl_time+3      -> pause 3 s
                #   ...        .. 2*swirl_time+3    -> -5 rev
                #   ...        .. 2*swirl_time+6    -> pause 3 s
                if t_pump < swirl_time:
                    jt.velocity[3] = pump_speed
                    mode = "Pump +5 rev"
                elif t_pump < swirl_time + 3.0:
                    jt.velocity[3] = 0.0
                    mode = "Pump pause after +5 rev"
                elif t_pump < 2 * swirl_time + 3.0:
                    jt.velocity[3] = -pump_speed
                    mode = "Pump -5 rev"
                elif t_pump < 2 * swirl_time + 6.0:
                    jt.velocity[3] = 0.0
                    mode = "Pump pause after -5 rev"
                else:
                    jt.velocity[3] = 0.0
                    mode = "Pump sequence done"
                    # Optional: auto-exit TEST_MOTOR here
                    # self._enter(Phase.INIT_POS)
                    # self.arm_pub.publish(jt)
                    # return

            # Publish pump command (still sent for compatibility)
            self.pump_pub.publish(pump_msg)

            # Send joint targets to I2C bridge
            self.arm_pub.publish(jt)

            # Log once per second to avoid spam
            if int(t) != int(t - 0.05):
                self.get_logger().info(
                    f"[TEST_MOTOR] seg={segment} mode={mode} "
                    f"dir={'+' if direction > 0 else '-'} "
                    f"pos={[round(a,3) for a in jt.position]} "
                    f"vel={[round(v,3) for v in jt.velocity]}"
                )

            return  # do not run the normal sequence below

        # ======== NORMAL SEQUENCE ========
        if self.phase == Phase.INIT_POS:
            if self._elapsed() >= self.t_settle:
                self._step_idx = 0
                self._start_step(self._step_idx)

        elif self.phase in (Phase.STEP0, Phase.STEP1, Phase.STEP2):
            self._run_step()

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
            self._publish_index(-1)
            self._enter(Phase.INIT_POS)

        elif self.phase == Phase.IDLE:
            pass

    # ---------- Step logic (index + pump timing) ----------
    def _start_step(self, step_idx: int):
        idx = self.order[step_idx]
        self._publish_index(idx)      # tell robothand which center to MOVE to
        # robothand will do MOVE → (settle) → SWIRL (+3cm)
        # we time the pump start/stop relative to when we set the index
        self._enter(Phase.STEP0 if step_idx == 0 else Phase.STEP1 if step_idx == 1 else Phase.STEP2)

    def _run_step(self):
        t = self._elapsed()

        # 1) wait for arm to reach the center (time-based for now)
        if (not self._did_start_pump) and t >= self.t_settle:
            # start pump (duration handled by PumpController or we stop explicitly below)
            self.pump.start(duty=1.0, duration_s=0.0)
            self._did_start_pump = True
            self.get_logger().info('Pump ON')

        # 2) stop pump after t_settle + t_pump
        if self._did_start_pump and t >= (self.t_settle + self.t_pump):
            self.pump.stop()
            self.get_logger().info('Pump OFF')

        # 3) allow robothand to finish its +3 cm lift ("SWIRL") window
        if t >= (self.t_settle + self.t_pump + self.t_swirl):
            # move to next step or camera
            if self.phase == Phase.STEP0:
                self._step_idx = 1
                self._start_step(self._step_idx)
            elif self.phase == Phase.STEP1:
                self._step_idx = 2
                self._start_step(self._step_idx)
            else:
                # finished STEP2 → return to init (-1) and do CAMERA
                self._publish_index(-1)
                self._enter(Phase.CAMERA)


def main():
    rclpy.init()
    node = StateNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
