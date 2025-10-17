import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Int32
from soes_msgs.msg import JointTargets, CupcakeCenters

class RoboHandNode(Node):
    def __init__(self):
        super().__init__('soes_robothand')
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.index_sub = self.create_subscription(Int32, '/state/active_index', self._on_index, 10)
        self.center_sub = self.create_subscription(CupcakeCenters, '/vision/centers', self._on_centers, qos)
        self.targets_pub = self.create_publisher(JointTargets, '/arm/joint_targets', 10)
        self.active_index = -1
        self.centers = None
        self.timer = self.create_timer(0.05, self._tick)  # 20 Hz
        self.get_logger().info('soes_robothand node started (stub).')

    def _on_index(self, msg: Int32):
        self.active_index = int(msg.data)

    def _on_centers(self, msg: CupcakeCenters):
        self.centers = [(p.x, p.y, p.z) for p in msg.centers]

    def _tick(self):
        if self.centers is None or self.active_index not in (0,1,2):
            return
        # dummy joint targets
        out = JointTargets()
        out.position = [0.0, 0.0, 0.0, 0.0]
        out.velocity = [0.0, 0.0, 0.0, 0.0]
        out.use_velocity = False
        self.targets_pub.publish(out)

def main():
    rclpy.init()
    node = RoboHandNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
