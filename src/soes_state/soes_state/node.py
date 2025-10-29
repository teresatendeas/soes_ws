import enum
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Int32
from soes_msgs.msg import PumpCmd, VisionQuality
from soes_msgs.srv import RollTray

from .utils import PumpController


class Phase(enum.Enum):
    INIT_POS    = 0
    STEP0       = 1   # index = order[0]
    STEP1       = 2   # index = order[1]
    STEP2       = 3   # index = order[2]
    CAMERA      = 4
    ROLL_TRAY   = 5
    IDLE        = 6


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
        if self.phase == Phase.INIT_POS:
            # After init settle, start STEP0 at order[0]
            if self._elapsed() >= self.t_settle:
                self._step_idx = 0
                self._start_step(self._step_idx)

        elif self.phase in (Phase.STEP0, Phase.STEP1, Phase.STEP2):
            self._run_step()

        elif self.phase == Phase.CAMERA:
            # wait up to camera_timeout for an updated quality decision
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

            # restart the cycle
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
