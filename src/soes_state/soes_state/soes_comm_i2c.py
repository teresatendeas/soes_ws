import struct
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from smbus2 import SMBus, i2c_msg
from soes_msgs.msg import PumpCmd, JointTargets

class I2CBridge(Node):
    def __init__(self):
        super().__init__('soes_comm_i2c')

        # Parameters aligned with ESP
        self.declare_parameter('i2c_bus', 1)
        self.declare_parameter('i2c_addr', 0x08)
        self.declare_parameter('pos_scale', 1000)   # matches ESP POS_SCALE
        self.declare_parameter('vel_scale', 1000)   # matches ESP VEL_SCALE
        self.declare_parameter('debug', True)

        self.bus_id = int(self.get_parameter('i2c_bus').value)
        self.addr   = int(self.get_parameter('i2c_addr').value)
        self.pos_s  = int(self.get_parameter('pos_scale').value)
        self.vel_s  = int(self.get_parameter('vel_scale').value)
        self.debug  = bool(self.get_parameter('debug').value)

        self.bus = SMBus(self.bus_id)

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(PumpCmd, '/pump/cmd', self.on_pump, qos)
        self.create_subscription(JointTargets, '/arm/joint_targets', self.on_joint, qos)

        self.get_logger().info(f'I2C bridge up (bus={self.bus_id}, addr=0x{self.addr:02X}).')

    def on_pump(self, msg: PumpCmd):
        on_u8 = 1 if msg.on else 0
        duty_u8 = int(max(0.0, min(1.0, msg.duty)) * 100)
        duration_ms = int(max(0.0, msg.duration_s) * 1000.0)
        frame = struct.pack('<BBBBH', 0x01, on_u8, duty_u8, 0x00, duration_ms)
        self._i2c_send_raw(frame)
        if self.debug:
            self.get_logger().info(f'I2C PUMP -> on={on_u8}, duty%={duty_u8}, dur_ms={duration_ms}')

    def on_joint(self, msg: JointTargets):
        pos = [float(msg.position[i]) if i < len(msg.position) else 0.0 for i in range(4)]
        vel = [float(msg.velocity[i]) if hasattr(msg, 'velocity') and i < len(msg.velocity) else 0.0 for i in range(4)]
        use_velocity = 1 if any(abs(v) > 1e-6 for v in vel) else 0

        def s16(x): 
            return max(-32768, min(32767, int(x)))

        pos_s16 = [s16(p * self.pos_s) for p in pos]      # radians * 1000
        vel_s16 = [s16(v * self.vel_s) for v in vel]      # rad/s * 1000

        frame = struct.pack('<BBhhhhhhhh',
                            0x03, use_velocity,
                            pos_s16[0], pos_s16[1], pos_s16[2], pos_s16[3],
                            vel_s16[0], vel_s16[1], vel_s16[2], vel_s16[3])
        self._i2c_send_raw(frame)
        if self.debug:
            self.get_logger().info(f'I2C 0x03 -> use_vel={use_velocity}, pos_s16={pos_s16}, vel_s16={vel_s16}')

    def _i2c_send_raw(self, frame: bytes):
        try:
            msg = i2c_msg.write(self.addr, frame)
            self.bus.i2c_rdwr(msg)
        except Exception as e:
            self.get_logger().warn(f'I2C send failed: {e}')

def main():
    rclpy.init()
    node = I2CBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
