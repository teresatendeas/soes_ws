import math
from typing import Optional, List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Int32
from soes_msgs.msg import JointTargets, CupcakeCenters  # using existing message names

class RoboHandNode(Node):
    def __init__(self):
        super().__init__('soes_robothand')

        # Parameters
        self.declare_parameter('control_rate_hz', 20.0)
        self.rate = float(self.get_parameter('control_rate_hz').value)

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)

        # Subscriptions
        self.index_sub = self.create_subscription(Int32, '/state/active_index', self._on_index, 10)
        self.centers_sub = self.create_subscription(CupcakeCenters, '/vision/centers', self._on_centers, qos)

        # Publisher
        self.targets_pub = self.create_publisher(JointTargets, '/arm/joint_targets', 10)

        # Runtime buffers
        self.active_index: int = -1
        self.centers: Optional[List[Tuple[float, float, float]]] = None

        # Timer to stream dummy joint targets
        self.timer = self.create_timer(1.0 / self.rate, self._on_timer)

        self.get_logger().info('soes_robothand started. Subscribed to /state/active_index and /vision/centers.')

    # --- callbacks ---
    def _on_index(self, msg: Int32):
        self.active_index = int(msg.data)
        self.get_logger().info(f'Active index updated: {self.active_index}')

    def _on_centers(self, msg: CupcakeCenters):
        self.centers = [(p.x, p.y, p.z) for p in msg.centers]

    # --- periodic publisher ---
    def _on_timer(self):
        # Only publish when we have a valid target
        if self.centers is None:
            return
        if self.active_index not in (0, 1, 2):
            return

        x, y, z = self.centers[self.active_index]

        # Dummy "IK": base yaw = atan2(y, x); other joints = 0
        q0 = math.atan2(y, x)
        q1, q2, q3 = 0.0, 0.0, 0.0

        msg = JointTargets()
        msg.position = [q0, q1, q2, q3]
        msg.velocity = [0.0, 0.0, 0.0, 0.0]
        msg.use_velocity = False
        self.targets_pub.publish(msg)
