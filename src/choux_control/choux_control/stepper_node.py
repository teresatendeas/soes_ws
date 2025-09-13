import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# >>> Ganti ini dengan library driver stepper kamu (misal RPi.GPIO, pigpio, atau serial ke driver)
# import RPi.GPIO as GPIO

class StepperNode(Node):
    def __init__(self):
        super().__init__('stepper_node')
        self.declare_parameter('steps_per_rev', 200)
        self.declare_parameter('rpm', 60)

        # Subscriber buat command
        self.subscription = self.create_subscription(
            String,
            'stepper_cmd',
            self.listener_callback,
            10
        )
        self.subscription

        self.get_logger().info("Stepper node started. Listening on /stepper_cmd")

    def listener_callback(self, msg):
        cmd = msg.data
        self.get_logger().info(f"Received command: {cmd}")
        if cmd == "extrude":
            self.extrude()
        elif cmd == "stop":
            self.stop()

    def extrude(self):
        rpm = self.get_parameter('rpm').value
        self.get_logger().info(f"Extruding at {rpm} RPM...")
        # TODO: masukkan kode GPIO/driver stepper

    def stop(self):
        self.get_logger().info("Stopping stepper motor")
        # TODO: matikan stepper

def main(args=None):
    rclpy.init(args=args)
    node = StepperNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

