import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Point
from soes_msgs.msg import CupcakeCenters, VisionQuality  # using existing message names

class VisionNode(Node):
    def __init__(self):
        super().__init__('soes_vision')

        # Parameters (no camera yet — just publish dummy data)
        self.declare_parameter('publish_rate_hz', 5.0)
        self.declare_parameter('frame_id', 'robot_base')
        # three targets in meters (x forward, y left, z up relative to base/camera)
        self.declare_parameter('centers_m', [0.22, 0.05, 0.10,   # left
                                             0.20, 0.00, 0.10,   # center
                                             0.22,-0.05, 0.10])  # right
        self.declare_parameter('diameter_mean_mm', [30.0, 30.0, 30.0])
        self.declare_parameter('quality_tolerance_mm', 3.0)  # if spread > tolerance -> needs_human

        self.rate = float(self.get_parameter('publish_rate_hz').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        arr = list(self.get_parameter('centers_m').value)
        self.centers = [
            (arr[0], arr[1], arr[2]),
            (arr[3], arr[4], arr[5]),
            (arr[6], arr[7], arr[8]),
        ]
        self.diam_mean = list(self.get_parameter('diameter_mean_mm').value)
        self.tol = float(self.get_parameter('quality_tolerance_mm').value)

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)

        self.centers_pub = self.create_publisher(CupcakeCenters, '/vision/centers', qos)
        self.quality_pub = self.create_publisher(VisionQuality, '/vision/quality', qos)

        self.k = 0
        self.timer = self.create_timer(1.0 / self.rate, self._on_timer)

        self.get_logger().info('soes_vision started. Publishing dummy centers & quality.')

    def _on_timer(self):
        now = self.get_clock().now().to_msg()

        # --- centers ---
        msg_c = CupcakeCenters()
        msg_c.header.stamp = now
        msg_c.header.frame_id = self.frame_id
        msg_c.frame_id = self.frame_id  # field exists in your msg
        for (x, y, z) in self.centers:
            p = Point(); p.x = float(x); p.y = float(y); p.z = float(z)
            msg_c.centers.append(p)
        self.centers_pub.publish(msg_c)

        # --- quality (dummy sizes that slowly vary) ---
        msg_q = VisionQuality()
        msg_q.header.stamp = now
        # little deterministic wiggle so state machine can react
        d0 = self.diam_mean[0] + 1.0*math.sin(self.k * 0.2)
        d1 = self.diam_mean[1] + 0.5*math.sin(self.k * 0.15 + 1.0)
        d2 = self.diam_mean[2] + 0.7*math.sin(self.k * 0.18 + 2.0)
        msg_q.diameter_mm = [float(d0), float(d1), float(d2)]
        msg_q.score = [1.0, 1.0, 1.0]
        spread = max(msg_q.diameter_mm) - min(msg_q.diameter_mm)
        msg_q.needs_human = bool(spread > self.tol)
        self.quality_pub.publish(msg_q)

        self.k += 1
