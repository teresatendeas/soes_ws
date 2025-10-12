import rclpy
from rclpy.node import Node

class RoboHandNode(Node):
    def __init__(self):
        super().__init__('soes_robothand')
        self.get_logger().info('soes_robothand node started (placeholder).')

def main():
    rclpy.init()
    node = RoboHandNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
