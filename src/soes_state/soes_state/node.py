#!/usr/bin/env python3
import enum
import math
from typing import Optional, List

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Bool, Int32
from soes_msgs.msg import (
    PumpCmd,
    VisionQuality,
    JointTargets,
    RollerCmd,
    CupcakeCenters,
)


# ---------------- High-level phases ----------------
class Phase(enum.Enum):
    INIT_POS   = 0
    STEP0      = 1
    STEP1      = 2
    STEP2      = 3
    CAMERA     = 4
    ROLL_TRAY  = 5
    IDLE       = 6
    TEST_MOTOR = 7


class StateNode(Node):
    def __init__(self):
        super().__init__("soes_state")

        # ---------- Parameters ----------
        # Timing cupcake process
        self.declare_parameter("settle_before_pump_s", 0.6)
        self.declare_parameter("pump_on_s", 2.0)
        self.declare_parameter("swirl_time_s", 1.0)
        self.declare_parameter("order", [0, 1, 2])  # urutan cup

        # Robot kinematics
        self.declare_parameter("link_l1_m", 0.090)
        self.declare_parameter("link_l2_m", 0.110)
        self.declare_parameter("link_l3_m", 0.080)
        self.declare_parameter("tool_offset_m", 0.030)

        # joint limits dan kecepatan default
        self.declare_parameter("q_home_deg", [0.0, -40.0, 80.0, 90.0])
        self.declare_parameter("max_joint_speed_deg_s", [40.0, 40.0, 40.0, 90.0])

        # Trajectory
        self.declare_parameter("traj_dt_s", 0.02)
        self.declare_parameter("s_curve_ratio", 0.3)

        # YOLO / vision
        self.declare_parameter("use_vision", True)
        self.declare_parameter("vision_timeout_s", 3.0)

        # Publikasi
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ---------- Publishers ----------
        self.pub_phase = self.create_publisher(Int32, "/state/phase", 10)
        self.pub_active_index = self.create_publisher(Int32, "/state/active_index", 10)

        # ke I2C bridge (harus cocok dengan I2CBridge.on_joint)
        self.pub_joint_targets = self.create_publisher(JointTargets, "/arm/joint_targets", qos)

        # ke I2C bridge (harus cocok dengan I2CBridge.on_pump / on_roller)
        self.pub_pump = self.create_publisher(PumpCmd, "/pump/cmd", qos)
        self.pub_roller = self.create_publisher(RollerCmd, "/roller/cmd", qos)

        # ---------- Subscribers ----------
        # UI / logic high-level
        self.sub_start = self.create_subscription(
            Bool, "/state/start", self._on_start, 10
        )
        self.sub_reset = self.create_subscription(
            Bool, "/state/reset", self._on_reset, 10
        )

        # Vision
        self.sub_vision_centers = self.create_subscription(
            CupcakeCenters, "/vision/centers", self._on_vision_centers, qos
        )
        self.sub_vision_quality = self.create_subscription(
            VisionQuality, "/vision/quality", self._on_vision_quality, qos
        )

        # Pause dari ESP (I2CBridge)
        self.sub_esp_paused = self.create_subscription(
            Bool, "/esp_paused", self._on_esp_paused, 10
        )

        # ---------- Internal state ----------
        self.phase: Phase = Phase.IDLE
        self.last_phase: Phase = Phase.IDLE

        self.active_index: int = -1      # cup index; -1 = HOME
        # ambil order dari parameter
        order_param = self.get_parameter("order").get_parameter_value()
        self.order: List[int] = list(order_param.integer_array_value) or [0, 1, 2]

        self._has_started: bool = False
        self._reset_requested: bool = False

        # flag pause dari ESP
        self.esp_paused: bool = False

        # robot joint state (rad)
        q_home_deg = self.get_parameter("q_home_deg").value
        self.q_home = np.radians(np.array(q_home_deg, dtype=float))
        self.q_current = self.q_home.copy()

        # Trajectory buffer (joint-space)
        self.traj: Optional[np.ndarray] = None
        self.traj_index: int = 0

        # Vision buffer
        self.latest_centers: Optional[CupcakeCenters] = None
        self.latest_quality: Optional[VisionQuality] = None
        self.last_vision_stamp = self.get_clock().now()

        # Timers
        self.dt = float(self.get_parameter("traj_dt_s").value)
        self.timer_main = self.create_timer(self.dt, self._on_timer)

        # Pump timing
        self._pump_on_until: Optional[rclpy.time.Time] = None
        self._settle_until: Optional[rclpy.time.Time] = None

        # Log awal
        self.get_logger().info("[STATE] Node soes_state (merged) started.")
        self._publish_phase()

    # ==================== Callbacks ====================

    def _on_start(self, msg: Bool):
        if not msg.data:
            return

        # Kalau masih jalan (bukan IDLE), jangan restart
        if self.phase != Phase.IDLE:
            self.get_logger().warn(
                f"[STATE] Start received but phase={self.phase.name} (not IDLE). Ignoring."
            )
            return

        self.get_logger().info("[STATE] Start sequence requested.")
        self._has_started = True
        self._reset_requested = False
        self.phase = Phase.INIT_POS
        self.active_index = -1
        self._plan_home()
        self._publish_phase()

    def _on_reset(self, msg: Bool):
        if not msg.data:
            return

        self.get_logger().warn("[STATE] Reset requested.")
        self._reset_requested = True
        self._has_started = False
        self.phase = Phase.IDLE
        self.active_index = -1

        # clear traj
        self.traj = None
        self.traj_index = 0

        # matikan aktuator
        self._pump_off()
        self._roller_off()

        self._publish_phase()

    def _on_vision_centers(self, msg: CupcakeCenters):
        self.latest_centers = msg
        self.last_vision_stamp = self.get_clock().now()

    def _on_vision_quality(self, msg: VisionQuality):
        self.latest_quality = msg
        self.last_vision_stamp = self.get_clock().now()

    def _on_esp_paused(self, msg: Bool):
        # hardware pause dari ESP
        self.esp_paused = msg.data
        if self.esp_paused:
            self.get_logger().warn("[STATE] ESP paused -> freezing state machine + traj")
        else:
            self.get_logger().info("[STATE] ESP resumed -> continuing state machine")

    # ==================== Main timer ====================

    def _on_timer(self):
        # Kalau ESP pause, freeze semua logic state / traj
        if self.esp_paused:
            return

        # update joint following traj
        self._step_trajectory()

        # step high-level phase machine
        if self.phase == Phase.IDLE:
            self._step_idle()
        elif self.phase == Phase.INIT_POS:
            self._step_init_pos()
        elif self.phase == Phase.STEP0:
            self._step_step_phase(0, Phase.STEP1)
        elif self.phase == Phase.STEP1:
            self._step_step_phase(1, Phase.STEP2)
        elif self.phase == Phase.STEP2:
            self._step_step_phase(2, Phase.CAMERA)
        elif self.phase == Phase.CAMERA:
            self._step_camera()
        elif self.phase == Phase.ROLL_TRAY:
            self._step_roll_tray()
        elif self.phase == Phase.TEST_MOTOR:
            self._step_test_motor()

    # ==================== Phase helpers ====================

    def _step_idle(self):
        # di IDLE, robot tetap di posisi sekarang
        pass

    def _step_init_pos(self):
        if not self._trajectory_active():
            self.get_logger().info("[STATE] Reached HOME. Go to STEP0.")
            self.phase = Phase.STEP0
            self._publish_phase()
            self.active_index = self.order[0]
            self._plan_move_to_cup(self.active_index)

    def _step_step_phase(self, order_idx: int, next_phase: Phase):
        if self._trajectory_active():
            return

        now = self.get_clock().now()

        # Sampai di cup, tunggu settle lalu nyalakan pump dan swirl
        if self._settle_until is None:
            settle_s = float(self.get_parameter("settle_before_pump_s").value)
            self._settle_until = now + Duration(seconds=settle_s)
            self.get_logger().info(f"[STATE] STEP{order_idx}: settling for pump.")
            return

        if now < self._settle_until:
            return

        # settle selesai, nyalakan pump dan swirl
        if self._pump_on_until is None:
            pump_s = float(self.get_parameter("pump_on_s").value)
            self._pump_on(now, pump_s)
            self._plan_swirl_about_current_cup()
            self.get_logger().info(f"[STATE] STEP{order_idx}: pump ON + swirl.")
            return

        # cek apakah pump sudah selesai
        if now < self._pump_on_until:
            return

        # Pump selesai
        self._pump_off()
        self._settle_until = None
        self._pump_on_until = None

        # Lanjut ke cup berikut atau ke next_phase
        if order_idx + 1 < len(self.order):
            self.phase = Phase(Phase.STEP0.value + order_idx + 1)
            self.active_index = self.order[order_idx + 1]
            self.get_logger().info(
                f"[STATE] STEP{order_idx}: done, go to STEP{order_idx+1} (cup idx={self.active_index})."
            )
            self._publish_phase()
            self._plan_move_to_cup(self.active_index)
        else:
            self.get_logger().info("[STATE] All steps done. Go to CAMERA.")
            self.phase = next_phase
            self._publish_phase()
            self.active_index = -1
            self._plan_home()

    def _step_camera(self):
        use_vision = bool(self.get_parameter("use_vision").value)
        if not use_vision:
            self.get_logger().info("[STATE] CAMERA: vision disabled, go to ROLL_TRAY.")
            self.phase = Phase.ROLL_TRAY
            self._publish_phase()
            return

        if self.latest_quality is None:
            return

        # sesuaikan field dengan definisi VisionQuality kamu
        if getattr(self.latest_quality, "ok", False):
            self.get_logger().info("[STATE] CAMERA: quality OK, go to ROLL_TRAY.")
            self.phase = Phase.ROLL_TRAY
            self._publish_phase()

    def _step_roll_tray(self):
        # RollerCmd di I2CBridge pakai field 'on' saja
        cmd = RollerCmd()
        cmd.on = True
        self.pub_roller.publish(cmd)

        self.get_logger().info("[STATE] ROLL_TRAY: send roller ON, then IDLE.")
        self.phase = Phase.IDLE
        self._publish_phase()

    def _step_test_motor(self):
        # Mode untuk test motor manual (kalau kamu pakai)
        pass

    # ==================== Trajectory + joint targets ====================

    def _trajectory_active(self) -> bool:
        return self.traj is not None and self.traj_index < len(self.traj)

    def _step_trajectory(self):
        if not self._trajectory_active():
            return

        q = self.traj[self.traj_index]
        self.q_current = q.copy()
        self.traj_index += 1

        # publish JointTargets ke I2C bridge
        msg = JointTargets()
        # I2CBridge.on_joint mengharapkan 'position' dan opsional 'velocity'
        msg.position = list(float(v) for v in q)
        msg.velocity = [0.0, 0.0, 0.0, 0.0]
        self.pub_joint_targets.publish(msg)

    def _plan_home(self):
        self._plan_joint_traj(self.q_current, self.q_home)

    def _plan_move_to_cup(self, idx: int):
        if self.latest_centers is None:
            self.get_logger().warn("[STATE] No centers from vision, using dummy center.")
            x, y, z = 0.18, 0.0, -0.13
        else:
            # Sesuaikan field dengan CupcakeCenters kamu
            if idx == 0:
                x, y, z = (
                    self.latest_centers.c1.x,
                    self.latest_centers.c1.y,
                    self.latest_centers.c1.z,
                )
            elif idx == 1:
                x, y, z = (
                    self.latest_centers.c2.x,
                    self.latest_centers.c2.y,
                    self.latest_centers.c2.z,
                )
            else:
                x, y, z = (
                    self.latest_centers.c3.x,
                    self.latest_centers.c3.y,
                    self.latest_centers.c3.z,
                )

        q_target = self._ik_cartesian_target(x, y, z)
        if q_target is None:
            self.get_logger().error(f"[STATE] IK failed for cup {idx}.")
            return

        self._plan_joint_traj(self.q_current, q_target)

    def _plan_swirl_about_current_cup(self):
        swirl_time = float(self.get_parameter("swirl_time_s").value)
        steps = max(1, int(swirl_time / self.dt))

        q_start = self.q_current.copy()
        q = np.zeros((steps, 4))
        for i in range(steps):
            t = i / max(steps - 1, 1)
            q[i, :] = q_start
            # putar joint-4 sekitar 180 derajat bolak balik
            q[i, 3] = q_start[3] + math.radians(180.0) * math.sin(math.pi * t)

        self.traj = q
        self.traj_index = 0

    def _plan_joint_traj(self, q_start: np.ndarray, q_goal: np.ndarray):
        max_speed_deg = np.array(self.get_parameter("max_joint_speed_deg_s").value)
        max_speed = np.radians(max_speed_deg)

        dq = np.abs(q_goal - q_start)
        t_needed = np.max(dq / np.maximum(max_speed, 1e-3))
        t_needed = max(t_needed, self.dt)
        steps = max(2, int(t_needed / self.dt))

        q = np.zeros((steps, 4))
        for i in range(steps):
            s = i / (steps - 1)
            s_smooth = s * s * (3.0 - 2.0 * s)
            q[i, :] = q_start + s_smooth * (q_goal - q_start)

        self.traj = q
        self.traj_index = 0

    # ==================== IK ====================

    def _ik_cartesian_target(self, x: float, y: float, z: float) -> Optional[np.ndarray]:
        l1 = float(self.get_parameter("link_l1_m").value)
        l2 = float(self.get_parameter("link_l2_m").value)
        l3 = float(self.get_parameter("link_l3_m").value)
        tool_offset = float(self.get_parameter("tool_offset_m").value)

        # yaw
        q0 = math.atan2(y, x)

        r = math.sqrt(x * x + y * y)
        z_eff = z + tool_offset

        # treat l2 + l3 as satu link
        L = l2 + l3

        D = (r * r + z_eff * z_eff - l1 * l1 - L * L) / (2.0 * l1 * L)
        if D < -1.0 or D > 1.0:
            self.get_logger().error(f"[IK] Unreachable: D={D:.3f}")
            return None

        q2 = math.acos(D)  # elbow
        alpha = math.atan2(z_eff, r)
        beta = math.atan2(L * math.sin(q2), l1 + L * math.cos(q2))
        q1 = alpha - beta

        # q3: servo topping center 90 deg
        q3_center_deg = 90.0
        q3 = math.radians(q3_center_deg)

        q = np.array([q0, q1, q2, q3], dtype=float)
        return q

    # ==================== Pump & roller ====================

    def _pump_on(self, now: rclpy.time.Time, duration_s: float):
        cmd = PumpCmd()
        cmd.on = True
        self.pub_pump.publish(cmd)
        self._pump_on_until = now + Duration(seconds=duration_s)

    def _pump_off(self):
        cmd = PumpCmd()
        cmd.on = False
        self.pub_pump.publish(cmd)
        self._pump_on_until = None

    def _roller_off(self):
        cmd = RollerCmd()
        cmd.on = False
        self.pub_roller.publish(cmd)

    # ==================== Utils ====================

    def _publish_phase(self):
        if self.phase != self.last_phase:
            self.get_logger().info(f"[STATE] Phase -> {self.phase.name}")
            self.last_phase = self.phase

        msg = Int32()
        msg.data = int(self.phase.value)
        self.pub_phase.publish(msg)

        idx_msg = Int32()
        idx_msg.data = int(self.active_index)
        self.pub_active_index.publish(idx_msg)


def main(args=None):
    rclpy.init(args=args)
    node = StateNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
