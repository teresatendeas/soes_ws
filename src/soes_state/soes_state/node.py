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
    POS1_PUMP   = 1
    POS2_PUMP   = 2
    POS3_PUMP   = 3
    RETURN_INIT = 4
    CAMERA      = 5
    ROLL_TRAY   = 6
    IDLE        = 7

class StateNode(Node):
    def __init__(self):
        super().__init__('soes_state')

        # ----- parameters (tune in YAML) -----
        self.declare_parameter('pump_on_s', 2.0)         # pump ON duration at each position
        self.declare_parameter('settle_s', 0.5)          # settle time after index change
        self.declare_parameter('order', [1, 0, 2])       # index order for POS1, POS2, POS3
        self.declare_parameter('roller_distance_mm', 100.0)
        self.declare_parameter('roller_speed_mm_s', 40.0)
        self.declare_parameter('camera_timeout_s', 2.0)  # wait time in CAMERA state

        self.pump_on_s  = float(self.get_parameter('pump_on_s').value)
        self.settle_s   = float(self.get_parameter('settle_s').value)
        self.order      = list(self.get_parameter('order').value)
        self.roll_dist  = float(self.get_parameter('roller_distance_mm').value)
        self.roll_speed = float(self.get_parameter('roller_speed_mm_s').value)
        self.cam_to     = float(self.get_parameter('camera_timeout_s').value)

        qos = QoSProfile(depth=10,
                         reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)

        # ----- pubs/subs/srvs -----
        self.index_pub = self.create_publisher(Int32, '/state/active_index', 1)  # 0,1,2 or -1 for init
        self.pump_pub  = self.create_publisher(PumpCmd, '/pump/cmd', 1)
        self.qual_sub  = self.create_subscription(VisionQuality, '/vision/quality', self.on_quality, qos)
        self.roll_cli  = self.create_client(RollTray, '/tray/roll')

        # pump helper
        self.pump = PumpController(self._pump_on, self._pump_off)

        # runtime vars
        self.phase = Phase.INIT_POS
        self.phase_time = self.get_clock().now()
        self.quality_flag = False
        self._pos_step = 0           # 0→POS1, 1→POS2, 2→POS3
        self._pump_started = False

        # 20 Hz tick
        self.timer = self.create_timer(0.05, self.tick)
        self.get_logger().info('soes_state: sequence ready (INIT_POS).')

        # enter INIT_POS: set index to -1 (home/init pose)
        self._publish_index(-1)

    # --------- helpers ----------
    def _enter(self, new_phase: Phase):
        self.phase = new_phase
        self.phase_time = self.get_clock().now()
        self._pump_started = False
        self.get_logger().info(f'[STATE] -> {self.phase.name}')

    def _elapsed(self) -> float:
        return (self.get_clock().now() - self.phase_time).nanoseconds * 1e-9

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

    # --------- callbacks ----------
    def on_quality(self, msg: VisionQuality):
        self.quality_flag = bool(msg.needs_human)

    # --------- state machine tick ----------
    def tick(self):
        if self.phase == Phase.INIT_POS:
            # wait settle, then go to POS1
            self.get_logger().info('System Starting ...')
            if self._elapsed() >= self.settle_s:
                self._pos_step = 0
                self._enter(Phase.POS1_PUMP)
                self._publish_index(self.order[0])

        elif self.phase == Phase.POS1_PUMP:
            self.get_logger().info('Going to POS1 ...')
            self._handle_pump_phase(next_phase=Phase.POS2_PUMP, next_index=self.order[1])

        elif self.phase == Phase.POS2_PUMP:
            self.get_logger().info('Going to POS2 ...')
            self._handle_pump_phase(next_phase=Phase.POS3_PUMP, next_index=self.order[2])

        elif self.phase == Phase.POS3_PUMP:
            # last pump, then return to init
            self.get_logger().info('Going to POS3 ...')
            if not self._pump_started and self._elapsed() >= self.settle_s:
                self.pump.start(duty=1.0, duration_s=0.0)
                self._pump_started = True
            if self._pump_started and self._elapsed() >= (self.settle_s + self.pump_on_s):
                self.pump.stop()
                self._enter(Phase.RETURN_INIT)
                self._publish_index(-1)

        elif self.phase == Phase.RETURN_INIT:
            self.get_logger().info('Returning to init ...')
            if self._elapsed() >= self.settle_s:
                self._enter(Phase.CAMERA)

        elif self.phase == Phase.CAMERA:
            # wait up to camera_timeout for an updated quality message
            self.get_logger().info('Camera recording ...')
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
            self._enter(Phase.INIT_POS)
            self._publish_index(-1)

        elif self.phase == Phase.IDLE:
            pass

    def _handle_pump_phase(self, next_phase: Phase, next_index: int):
        # generic handler for POS1_PUMP and POS2_PUMP
        if not self._pump_started and self._elapsed() >= self.settle_s:
            self.pump.start(duty=1.0, duration_s=0.0)
            self._pump_started = True
        if self._pump_started and self._elapsed() >= (self.settle_s + self.pump_on_s):
            self.pump.stop()
            self._enter(next_phase)
            self._publish_index(next_index)

def main():
    rclpy.init()
    node = StateNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
