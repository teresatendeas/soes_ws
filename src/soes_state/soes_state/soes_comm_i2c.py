import struct
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from smbus2 import SMBus
from soes_msgs.msg import PumpCmd, JointTargets

class I2CBridge(Node):
    """
    Bridges ROS 2 topics to an ESP device via I²C (Jetson Nano SDA/SCL).
    Subscriptions:
      - /pump/cmd         (soes_msgs/PumpCmd)
      - /arm/joint_targets(soes_msgs/JointTargets)
    Sends compact binary frames to ESP32 I²C slave.
    """
    def __init__(self):
        super().__init__('soes_comm_i2c')

        # Parameters (loaded from soes_bringup/config/comm.yaml)
        self.declare_parameter('i2c_bus', 1)     # Jetson bus 1 => header pins SDA=3, SCL=5
        self.declare_parameter('i2c_addr', 0x28) # ESP32 slave address
        self.declare_parameter('deg_scale', 10)  # joints in degrees*10 (int16)
        self.declare_parameter('duty_scale', 100)# duty [0..1] -> [0..100]
        self.declare_parameter('debug', True)

        self.bus_id = int(self.get_parameter('i2c_bus').value)
        self.addr   = int(self.get_parameter('i2c_addr').value)
        self.deg_s  = int(self.get_parameter('deg_scale').value)
        self.duty_s = int(self.get_parameter('duty_scale').value)
        self.debug  = bool(self.get_parameter('debug').value)

        try:
            self.bus = SMBus(self.bus_id)
        except Exception as e:
            self.get_logger().error(f'Cannot open I2C bus {self.bus_id}: {e}')
            raise

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(PumpCmd, '/pump/cmd', self.on_pump, qos)
        self.create_subscription(JointTargets, '/arm/joint_targets', self.on_joint, qos)

        self.get_logger().info(f'I2C bridge up (bus={self.bus_id}, addr=0x{self.addr:02X}).')

    # Frames (little-endian, keep <=32 bytes):
    #  0x01 Pump:   [0x01, on(u8), duty(u8), rsv(u8), duration_ms(u16)]
    #  0x02 Joints: [0x02, q0(int16), q1(int16), q2(int16), q3(int16)]  # degrees*scale as int16

    def on_pump(self, msg: PumpCmd):
        on_u8 = 1 if msg.on else 0
        duty_u8 = int(max(0.0, min(1.0, msg.duty)) * self.duty_s)
        duration_ms = int(max(0.0, msg.duration_s) * 1000.0)
        frame = struct.pack('<BBBBH', 0x01, on_u8, duty_u8, 0x00, duration_ms)
        self._i2c_send(0x01, list(frame))
        if self.debug:
            self.get_logger().info(f'I2C PUMP -> on={on_u8}, duty%={duty_u8}, dur_ms={duration_ms}')

    def on_joint(self, msg: JointTargets):
        q_deg_scaled = []
        for i in range(4):
            rad = float(msg.position[i]) if i < len(msg.position) else 0.0
            deg_scaled = int(math.degrees(rad) * self.deg_s)
            deg_scaled = max(-32768, min(32767, deg_scaled))
            q_deg_scaled.append(deg_scaled)
        frame = struct.pack('<Bhhhh', 0x02, q_deg_scaled[0], q_deg_scaled[1], q_deg_scaled[2], q_deg_scaled[3])
        self._i2c_send(0x02, list(frame))
        if self.debug:
            self.get_logger().info(f'I2C JOINTS -> {q_deg_scaled}')

    def _i2c_send(self, command, data_bytes):
        try:
            payload = data_bytes[1:] if len(data_bytes) > 1 else []
            self.bus.write_i2c_block_data(self.addr, command, payload)
        except Exception as e:
            self.get_logger().warn(f'I2C send failed (cmd=0x{command:02X}): {e}')

def main():
    rclpy.init()
    node = I2CBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
