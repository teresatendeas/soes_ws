#!/usr/bin/env python3
import struct
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from smbus2 import SMBus, i2c_msg

from soes_msgs.msg import PumpCmd, JointTargets, RollerCmd
from std_msgs.msg import Bool

STATUS_SWITCH_BIT = 0x01  # bit 0 in ESP32 status byte = switch ON/OFF
STATUS_PAUSE_BIT  = 0x02  # bit 1 in ESP32 status byte = PAUSE
STATUS_SOES_BIT   = 0x04  # bit 2 in ESP32 status byte = SOES_DONE (new)

# New command: SOES status from Jetson -> ESP
CMD_SOES_STATUS = 0x20


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

        # I2C bus
        self.bus = SMBus(self.bus_id)

        # QoS for topics
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        # Subscriptions (Jetson -> ESP writes)
        self.create_subscription(PumpCmd, '/pump/cmd', self.on_pump, qos)
        self.create_subscription(JointTargets, '/arm/joint_targets', self.on_joint, qos)
        self.create_subscription(RollerCmd, '/roller/cmd', self.on_roller, qos)

        # Subscribe to SOES decision published by VisionNode
        # Topic: /vision/soes_done (Bool). When False -> needs human -> trigger LED+buzzer on ESP
        self.create_subscription(Bool, '/vision/soes_done', self.on_soes_done, qos)

        # Publisher for ESP switch status (ESP -> Jetson reads)
        self.switch_pub = self.create_publisher(Bool, '/esp_switch_on', 10)
        self.last_switch_state = None

        # Publisher for ESP pause state
        self.pause_pub = self.create_publisher(Bool, '/esp_paused', 10)
        self.last_pause_state = None

        # Local pause flag to gate outgoing frames
        self.paused = False

        # Timer to poll ESP status over I2C (e.g. 10 Hz)
        self.status_timer = self.create_timer(0.1, self.poll_status)

        self.get_logger().info(
            f'I2C bridge up (bus={self.bus_id}, addr=0x{self.addr:02X}).'
        )

    # -------------------------------------------------------------------------
    #  Pump command (write-only)
    # -------------------------------------------------------------------------
    def on_pump(self, msg: PumpCmd):
        # If hardware is paused, ignore pump commands
        if self.paused:
            if self.debug:
                self.get_logger().debug('ESP paused -> skipping pump command')
            return

        on_u8 = 1 if msg.on else 0

        frame = struct.pack('<BB', 0x10, on_u8)  # CMD_PUMP = 0x10
        self._i2c_send_raw(frame)

        if self.debug:
            self.get_logger().info(f'I2C PUMP (0x10) -> on={on_u8}')

    # -------------------------------------------------------------------------
    #  Roller command (write-only)
    # -------------------------------------------------------------------------
    def on_roller(self, msg: RollerCmd):
        # If hardware is paused, ignore roller commands
        if self.paused:
            if self.debug:
                self.get_logger().debug('ESP paused -> skipping roller command')
            return

        on_u8 = 1 if msg.on else 0

        frame = struct.pack('<BB', 0x11, on_u8)  # CMD_ROLLER = 0x11
        self._i2c_send_raw(frame)

        if self.debug:
            self.get_logger().info(f'I2C ROLLER (0x11) -> on={on_u8}')

    # -------------------------------------------------------------------------
    #  Joint command (write-only)
    # -------------------------------------------------------------------------
    def on_joint(self, msg: JointTargets):
        # If hardware is paused, do not send new joint frames
        if self.paused:
            if self.debug:
                self.get_logger().debug('ESP paused -> skipping joint frame')
            return

        pos = [float(msg.position[i]) if i < len(msg.position) else 0.0
               for i in range(4)]
        vel = [float(msg.velocity[i]) if hasattr(msg, 'velocity') and i < len(msg.velocity)
               else 0.0 for i in range(4)]

        # if any velocity is non-zero -> velocity mode
        use_velocity = 1 if any(abs(v) > 1e-6 for v in vel) else 0

        def s16(x):
            return max(-32768, min(32767, int(x)))

        pos_s16 = [s16(p * self.pos_s) for p in pos]  # radians * 1000
        vel_s16 = [s16(v * self.vel_s) for v in vel]  # rad/s * 1000

        # CMD 0x03 frame:
        # [cmd, use_velocity, 4 * int16 pos, 4 * int16 vel]
        frame = struct.pack(
            '<BBhhhhhhhh',
            0x03, use_velocity,
            pos_s16[0], pos_s16[1], pos_s16[2], pos_s16[3],
            vel_s16[0], vel_s16[1], vel_s16[2], vel_s16[3]
        )

        self._i2c_send_raw(frame)

        if self.debug:
            self.get_logger().info(
                f'I2C 0x03 -> use_vel={use_velocity}, '
                f'pos_s16={pos_s16}, vel_s16={vel_s16}'
            )

    # -------------------------------------------------------------------------
    #  SOES status from Vision (write-only)  <-- NEW
    # -------------------------------------------------------------------------
    def on_soes_done(self, msg: Bool):
        # If hardware is paused, ignore soes status commands
        if self.paused:
            if self.debug:
                self.get_logger().debug('ESP paused -> skipping soes status command')
            return

        soes_done_u8 = 1 if msg.data else 0
        frame = struct.pack('<BB', CMD_SOES_STATUS, soes_done_u8)
        self._i2c_send_raw(frame)

        if self.debug:
            self.get_logger().info(f'I2C SOES (0x20) -> soes_done={soes_done_u8}')

    # -------------------------------------------------------------------------
    #  Low-level I2C write
    # -------------------------------------------------------------------------
    def _i2c_send_raw(self, frame: bytes):
        try:
            msg = i2c_msg.write(self.addr, frame)
            self.bus.i2c_rdwr(msg)
        except Exception as e:
            self.get_logger().warn(f'I2C send failed: {e}')

    # -------------------------------------------------------------------------
    #  Poll ESP32 status byte over I2C (read)
    # -------------------------------------------------------------------------
    def poll_status(self):
        """
        Periodically called by a ROS2 timer.
        Reads 1 byte from ESP32 (onRequestI2C), decodes switch + pause state,
        and publishes /esp_switch_on and /esp_paused.
        """
        try:
            status = self.bus.read_byte(self.addr)
        except Exception as e:
            self.get_logger().warn(f'I2C read failed from 0x{self.addr:02X}: {e}')
            return

        # Decode bits
        switch_on = bool(status & STATUS_SWITCH_BIT)
        pause_on  = bool(status & STATUS_PAUSE_BIT)

        # Publish switch
        sw_msg = Bool()
        sw_msg.data = switch_on
        self.switch_pub.publish(sw_msg)

        # Publish pause
        pa_msg = Bool()
        pa_msg.data = pause_on
        self.pause_pub.publish(pa_msg)

        # Update local pause flag
        self.paused = pause_on

        # Edge-detection logging: switch
        if self.last_switch_state is None or switch_on != self.last_switch_state:
            if switch_on:
                self.get_logger().warn(
                    '[ESP SWITCH] ON -> restart / emergency request received!'
                )
            else:
                self.get_logger().info('[ESP SWITCH] OFF')
            self.last_switch_state = switch_on

        # Edge-detection logging: pause
        if self.last_pause_state is None or pause_on != self.last_pause_state:
            if pause_on:
                self.get_logger().warn('[ESP PAUSE] PAUSED -> hardware frozen')
            else:
                self.get_logger().info('[ESP PAUSE] RESUMED -> hardware running')
            self.last_pause_state = pause_on


def main():
    rclpy.init()
    node = I2CBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down soes_comm_i2c')
        try:
            node.bus.close()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
