#!/usr/bin/env python3 
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Point
from soes_msgs.msg import CupcakeCenters, VisionQuality

import cv2
import numpy as np
from std_msgs.msg import Bool

# ===== YOLO (Ultralytics) support =====
try:
    from ultralytics import YOLO as UltralyticsYOLO
    _HAS_YOLO = True
except Exception as e:
    UltralyticsYOLO = None
    _HAS_YOLO = False

_YOLO_MODEL = None  # global cached model


def load_yolo_model(
    model_path: str = "/home/jetson/soes_ws/src/soes_vision/weights/best.pt"
):
    global _YOLO_MODEL
    if not _HAS_YOLO:
        print("[YOLO] Ultralytics not available, using color-based fallback.")
        return None

    if _YOLO_MODEL is not None:
        print("[YOLO] Model already loaded, reusing cached model.")
        return _YOLO_MODEL

    print("[YOLO] Loading model (this may take a few seconds)...")
    try:
        _YOLO_MODEL = UltralyticsYOLO(model_path)
        print(f"[YOLO] Model loaded successfully from: {model_path}")
    except Exception as e:
        print(f"[YOLO] Failed to load model: {e}")
        _YOLO_MODEL = None

    return _YOLO_MODEL


class VisionNode(Node):
    def __init__(self):
        super().__init__('soes_vision')

        # parameters
        self.declare_parameter('publish_rate_hz', 5.0)
        self.declare_parameter('frame_id', 'robot_base')
        self.declare_parameter('centers_m', [0.22, 0.05, 0.10,
                                             0.20, 0.00, 0.10,
                                             0.22, -0.05, 0.10])
        self.declare_parameter('diameter_mean_mm', [30.0, 30.0, 30.0])
        self.declare_parameter('quality_tolerance_mm', 3.0)

        self.declare_parameter('camera_index', 0)
        self.declare_parameter('px_to_mm_ref', 0.1)

        self.rate = float(self.get_parameter('publish_rate_hz').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        arr = list(self.get_parameter('centers_m').value)
        self.centers = [
            (arr[0], arr[1], arr[2]),
            (arr[3], arr[4], arr[5]),
            (arr[6], arr[7], arr[8])
        ]
        self.diam_mean = list(self.get_parameter('diameter_mean_mm').value)
        self.tol = float(self.get_parameter('quality_tolerance_mm').value)

        self.cam_index = int(self.get_parameter('camera_index').value)
        self.px_to_mm_ref = float(self.get_parameter('px_to_mm_ref').value)

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        self.centers_pub = self.create_publisher(
            CupcakeCenters, '/vision/centers', qos
        )
        self.quality_pub = self.create_publisher(
            VisionQuality, '/vision/quality', qos
        )
        self.soess_done_pub = self.create_publisher(
            Bool, '/vision/soes_done', qos
        )

        self.request_sub = self.create_subscription(
            Bool, '/vision/request', self._on_request, qos
        )

        # ---------- Kamera: buka sekali ----------
        self.cap = cv2.VideoCapture(self.cam_index)
        if not self.cap.isOpened():
            self.get_logger().error(
                f"Camera index {self.cam_index} failed to open!"
            )
            self.cap = None
        else:
            self.get_logger().info("Camera opened ONCE at startup.")

        # ---------- YOLO: load sekali di awal ----------
        self.get_logger().info("YOLO: starting initial model load...")
        load_yolo_model()
        self.get_logger().info("YOLO: initial model load done.")

        # Dummy k untuk centers (kalau mau animasi dikit)
        self.k = 0

        # Timer hanya untuk publish CENTERS / info statis
        self.timer = self.create_timer(
            max(0.001, 1.0 / self.rate),
            self._on_timer
        )
        self.get_logger().info('soes_vision started.')

    def _on_timer(self):
        now = self.get_clock().now().to_msg()

        # centers message (boleh tetap dummy / statis)
        msg_c = CupcakeCenters()
        msg_c.header.stamp = now
        msg_c.header.frame_id = self.frame_id
        msg_c.frame_id = self.frame_id
        for (x, y, z) in self.centers:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = float(z)
            msg_c.centers.append(p)
        self.centers_pub.publish(msg_c)

        # Tidak publish VisionQuality atau soes_done di timer lagi
        self.k += 1

    def _on_request(self, msg: Bool):
        self.get_logger().info("VISION REQUEST: running YOLO on one frame...")

        # Pastikan model sudah ada
        model = load_yolo_model()
        if model is None:
            self.get_logger().warn(
                "YOLO model not available, you may want to use fallback here."
            )

        # Pastikan kamera ready
        if self.cap is None or not self.cap.isOpened():
            self.get_logger().error("Camera not opened, cannot capture frame.")
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.get_logger().error("Failed to read frame from camera.")
            return

        # Run detection (YOLO atau fallback)
        vis, good_cnts, yolo_labels = detect_choux_from_frame(frame)

        # Estimasi diameter per cupcake
        diam_mm = []
        for i in range(len(self.diam_mean)):
            if i < len(yolo_labels):
                _, cx, cy, bw, bh = yolo_labels[i]
                bw = max(1e-6, float(bw))
                est = float(self.diam_mean[i]) * (bw / max(1e-6, self.px_to_mm_ref))
                diam_mm.append(est)
            else:
                diam_mm.append(float(self.diam_mean[i]))

        # Quality message
        msg_q = VisionQuality()
        msg_q.header.stamp = self.get_clock().now().to_msg()
        msg_q.diameter_mm = [float(x) for x in diam_mm]
        msg_q.score = [1.0] * len(diam_mm)
        msg_q.needs_human = (
            max(msg_q.diameter_mm) - min(msg_q.diameter_mm)
        ) > self.tol

        self.quality_pub.publish(msg_q)

        # soes_done = True kalau tidak perlu human
        soes_done_msg = Bool()
        soes_done_msg.data = (not msg_q.needs_human)
        self.soess_done_pub.publish(soes_done_msg)

        if msg_q.needs_human:
            self.get_logger().warn('VISION (on-request): needs_human == True')
        else:
            self.get_logger().info('VISION (on-request): needs_human == False')


# ===== helper dan main sama seperti file kamu =====

def draw_detected(img, cnts, color=(0, 255, 0)):
    out = img.copy()
    for i, c in enumerate(cnts, 1):
        (cx, cy), r = cv2.minEnclosingCircle(c)
        center, r = (int(cx), int(cy)), int(r)
        cv2.circle(out, center, r, color, 3)
        cv2.putText(out, f"Choux {i}", (center[0] - 40, center[1] - r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def _detect_choux_color_fallback(img):
    # isi sesuai file kamu sebelumnya
    ...
    return vis, good, yolo_labels


def detect_choux_from_frame(img):
    # isi sesuai file kamu sebelumnya
    ...
    return vis, good, yolo_labels


def debug_detect_choux_from_usb(cam_index=0):
    # isi sesuai file kamu sebelumnya
    ...


def main():
    rclpy.init()
    node = VisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
