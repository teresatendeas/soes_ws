import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from soes_msgs.msg import JointTargets

class OneMotorCmd(Node):
    """
    Publishes /arm/joint_targets with only one joint set.
    Params:
      joint_index (int)      0..3 (which joint to move)
      target_deg (float)     target position in degrees
      use_velocity (bool)    send velocity instead of position
      velocity_deg_s (float) velocity value (deg/s) if use_velocity=true
      hold_s (float)         seconds to keep publishing (0 = one-shot)
      rate_hz (float)        publish rate when holding
    """
    def __init__(self):
        super().__init__('one_motor_cmd')
        self.declare_parameter('joint_index', 0)
        self.declare_parameter('target_deg', 30.0)
        self.declare_parameter('use_velocity', False)
        self.declare_parameter('velocity_deg_s', 0.0)
        self.declare_parameter('hold_s', 0.0)
        self.declare_parameter('rate_hz', 10.0)

        self.idx = int(self.get_parameter('joint_index').value)
        self.target = float(self.get_parameter('target_deg').value)
        self.use_vel = bool(self.get_parameter('use_velocity').value)
        self.vel = float(self.get_parameter('velocity_deg_s').value)
        self.hold_s = float(self.get_parameter('hold_s').value)
        self.rate = max(1e-3, float(self.get_parameter('rate_hz').value))

        if not (0 <= self.idx <= 3):
            self.get_logger().error('joint_index must be 0..3')
            rclpy.shutdown()
            return

        self.pub = self.create_publisher(JointTargets, '/arm/joint_targets', 10)

        self.msg = JointTargets()
        self.msg.position = [0.0, 0.0, 0.0, 0.0]
        self.msg.velocity = [0.0, 0.0, 0.0, 0.0]
        self.msg.position[self.idx] = self.target
        self.msg.use_velocity = bool(self.use_vel)
        if self.msg.use_velocity:
            self.msg.velocity[self.idx] = self.vel

        if self.hold_s <= 0.0:
            self.pub.publish(self.msg)
            self.get_logger().info(
                f'One-shot sent: joint[{self.idx}] -> {self.target} deg '
                f'(use_velocity={self.msg.use_velocity}, vel={self.vel})'
            )
            self.create_timer(0.15, self._shutdown_once)
        else:
            self.end_time = self.get_clock().now() + Duration(seconds=self.hold_s)
            self.timer = self.create_timer(1.0 / self.rate, self._tick)
            self.get_logger().info(f'Publishing for {self.hold_s:.2f}s at {self.rate:.2f} Hz...')

    def _tick(self):
        self.pub.publish(self.msg)
        if self.get_clock().now() >= self.end_time:
            self.get_logger().info('Done holding. Stopping.')
            self.timer.cancel()
            self._shutdown_once()

    def _shutdown_once(self):
        try:
            rclpy.shutdown()
        except Exception:
            pass

def main():
    rclpy.init()
    node = OneMotorCmd()
    if rclpy.ok():
        rclpy.spin(node)

if __name__ == '__main__':
    main()
