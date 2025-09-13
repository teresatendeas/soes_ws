import time
import json
from threading import Lock
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Bool, Float32, String

try:
    import serial
except Exception:
    serial = None


class SerialBackend:
    """
    Serial protocol (sederhana):
      EN <0/1>\n
      RATE <mlps>\n
      CMD <ml>\n   # one-shot extrude amount
      STOP\n
    MCU harus implementasi sendiri.
    """
    def __init__(self, port:str, baud:int, logger):
        if serial is None:
            raise RuntimeError("pyserial tidak tersedia. `pip install pyserial`")
        self.log = logger
        self.ser = serial.Serial(port=port, baudrate=baud, timeout=1)
        time.sleep(0.2)

    def enable(self, on:bool):
        self._send(f"EN {1 if on else 0}")

    def set_rate(self, mlps:float):
        self._send(f"RATE {mlps:.6f}")

    def command_ml(self, ml:float):
        self._send(f"CMD {ml:.6f}")

    def stop(self):
        self._send("STOP")

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass

    def _send(self, s:str):
        msg = (s+"\n").encode("ascii")
        self.ser.write(msg)
        self.ser.flush()


class SimBackend:
    """
    Simulasi tanpa hardware: hanya menghitung waktu & total_ml.
    """
    def __init__(self, logger):
        self.log = logger
        self.enabled = False
        self.rate_mlps = 0.0
        self.total_ml = 0.0
        self._last_t = time.time()

    def tick(self):
        t = time.time()
        dt = t - self._last_t
        self._last_t = t
        if self.enabled and self.rate_mlps > 0.0:
            self.total_ml += self.rate_mlps * dt

    def enable(self, on:bool):
        self.enabled = on

    def set_rate(self, mlps:float):
        self.rate_mlps = max(0.0, mlps)

    def command_ml(self, ml:float):
        # Jalankan dengan cara sederhana: nyalakan sebentar sesuai ml/rate
        if self.rate_mlps <= 0.0:
            raise RuntimeError("rate_mlps harus > 0 sebelum command_ml")
        dur = ml / self.rate_mlps
        self.enable(True)
        time.sleep(max(0.0, dur))
        self.enable(False)

    def stop(self):
        self.enable(False)

    def close(self):
        pass


class ExtruderNode(Node):
    def __init__(self):
        super().__init__('extruder_node')

        # -------- Parameters --------
        self.declare_parameter('backend', 'sim')  # 'sim' | 'serial'
        self.declare_parameter('serial_port', '/dev/ttyUSB0')
        self.declare_parameter('serial_baud', 115200)
        self.declare_parameter('max_rate_mlps', 3.0)
        self.declare_parameter('control_hz', 100.0)

        self.backend_name = self.get_parameter('backend').get_parameter_value().string_value
        self.serial_port = self.get_parameter('serial_port').get_parameter_value().string_value
        self.serial_baud = int(self.get_parameter('serial_baud').value)
        self.max_rate = float(self.get_parameter('max_rate_mlps').value)
        self.ctrl_hz = float(self.get_parameter('control_hz').value)

        # -------- Backend --------
        if self.backend_name == 'serial':
            self.backend = SerialBackend(self.serial_port, self.serial_baud, self.get_logger())
            self.get_logger().info(f'Using SERIAL backend on {self.serial_port} @ {self.serial_baud}')
        else:
            self.backend = SimBackend(self.get_logger())
            self.get_logger().info('Using SIM backend')

        # -------- State --------
        self._enabled = False
        self._rate_mlps = 0.0
        self._total_ml = 0.0
        self._lock = Lock()
        self._last_tick = time.time()

        # -------- Topics --------
        self.sub_enable = self.create_subscription(Bool, '/extruder/enable', self.cb_enable, 10)
        self.sub_rate   = self.create_subscription(Float32, '/extruder/rate_mlps', self.cb_rate, 10)
        self.sub_cmdml  = self.create_subscription(Float32, '/extruder/command_ml', self.cb_cmdml, 10)

        self.pub_total  = self.create_publisher(Float32, '/extruder/telemetry/total_ml', 10)
        self.pub_state  = self.create_publisher(String,  '/extruder/state', 10)

        # -------- Control timer --------
        self.timer = self.create_timer(1.0/self.ctrl_hz, self.on_timer)

        self.get_logger().info('Extruder node started.')

    # ---- Callbacks ----
    def cb_enable(self, msg:Bool):
        with self._lock:
            self._enabled = bool(msg.data)
            self.backend.enable(self._enabled)
            self.get_logger().info(f'Enable -> {self._enabled}')

    def cb_rate(self, msg:Float32):
        rate = max(0.0, float(msg.data))
        if rate > self.max_rate:
            rate = self.max_rate
            self.get_logger().warn(f'rate_mlps dibatasi ke max_rate={self.max_rate}')
        with self._lock:
            self._rate_mlps = rate
            self.backend.set_rate(rate)

    def cb_cmdml(self, msg:Float32):
        ml = max(0.0, float(msg.data))
        if ml == 0.0:
            return
        # One-shot ekstrusi sejumlah ml (blocking ringan di thread callback ini)
        try:
            if isinstance(self.backend, SimBackend):
                if self._rate_mlps <= 0.0:
                    self.get_logger().warn('Set rate_mlps > 0 sebelum command_ml.')
                    return
                self.backend.command_ml(ml)
                with self._lock:
                    self._total_ml += ml
            else:
                self.backend.command_ml(ml)
        except Exception as e:
            self.get_logger().error(f'command_ml error: {e}')

    # ---- Periodic ----
    def on_timer(self):
        # Update simulasi & total
        if isinstance(self.backend, SimBackend):
            self.backend.tick()
            # Sinkronkan total dengan backend sim
            with self._lock:
                self._total_ml = self.backend.total_ml

        # Publish telemetry
        self.pub_total.publish(Float32(data=float(self._total_ml)))
        state = {
            "enabled": self._enabled,
            "rate_mlps": self._rate_mlps,
            "max_rate_mlps": self.max_rate,
            "backend": self.backend_name,
            "total_ml": self._total_ml,
        }
        self.pub_state.publish(String(data=json.dumps(state)))

    def destroy_node(self):
        try:
            self.backend.stop()
            self.backend.close()
        except Exception:
            pass
        return super().destroy_node()


def main():
    rclpy.init()
    node = ExtruderNode()
    try:
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
