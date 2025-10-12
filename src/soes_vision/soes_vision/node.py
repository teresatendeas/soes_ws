import rclpy
from rclpy.node import Node

class VisionNode(Node):
    def __init__(self):
        super().__init__('soes_vision')
        self.get_logger().info('soes_vision node started (placeholder).')

def main():
    rclpy.init()
    node = VisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
