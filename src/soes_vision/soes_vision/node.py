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
except Exception:
    UltralyticsYOLO = None
    _HAS_YOLO = False

_YOLO_MODEL = None  # global cached model


def load_yolo_model(
    model_path: str = "/home/jetson/soes_ws/src/soes_vision/weights/best.pt"
):
    """Load YOLO model sekali, cache di global."""
    global _YOLO_MODEL
    if not _HAS_YOLO:
        print("[YOLO] Ultralytics not available, using color-based fallback.")
        return None

    if _YOLO_MODEL is not None:
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

        # -------- parameters --------
        self.declare_parameter('publish_rate_hz', 5.0)
        self.declare_parameter('frame_id', 'robot_base')
        self.declare_parameter('centers_m', [0.22, 0.05, 0.10,
                                             0.20, 0.00, 0.10,
                                             0.22, -0.05, 0.10])
        self.declare_parameter('diameter_mean_mm', [30.0, 30.0, 30.0])
        self.declare_parameter('quality_tolerance_mm', 3.0)

        self.declare_parameter('camera_index', 0)
        self.declare_parameter('px_to_mm_ref', 0.1)

        # optional: tampilan debug
        self.declare_parameter('visualize', False)

        self.rate = float(self.get_parameter('publish_rate_hz').value)
        self.frame_id = str(self.get_parameter('frame_id').value)

        arr = list(self.get_parameter('centers_m').value)
        self.centers = [
            (arr[0], arr[1], arr[2]),
            (arr[3], arr[4], arr[5]),
            (arr[6], arr[7], arr[8]),
        ]

        self.diam_mean = list(self.get_parameter('diameter_mean_mm').value)
        self.tol = float(self.get_parameter('quality_tolerance_mm').value)

        self.cam_index = int(self.get_parameter('camera_index').value)
        self.px_to_mm_ref = float(self.get_parameter('px_to_mm_ref').value)
        self.visualize = bool(self.get_parameter('visualize').value)

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        # -------- publishers --------
        self.centers_pub = self.create_publisher(
            CupcakeCenters, '/vision/centers', qos
        )
        self.quality_pub = self.create_publisher(
            VisionQuality, '/vision/quality', qos
        )
        self.soess_done_pub = self.create_publisher(
            Bool, '/vision/soes_done', qos
        )

        # -------- subscriber --------
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

        # Dummy k (kalau mau animasi centers)
        self.k = 0

        # Timer hanya untuk publish CENTERS / info statis
        self.timer = self.create_timer(
            max(0.001, 1.0 / self.rate),
            self._on_timer
        )
        self.get_logger().info('soes_vision started.')

    def _on_timer(self):
        now = self.get_clock().now().to_msg()

        # publish centers (saat ini masih statis)
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

        self.k += 1

    def _on_request(self, msg: Bool):
        self.get_logger().info("VISION REQUEST: running YOLO on one frame...")

        # Pastikan model sudah ada
        model = load_yolo_model()
        if model is None:
            self.get_logger().warn(
                "YOLO model not available, using color-based fallback."
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

        # Debug window kalau visualize = True
        if self.visualize and vis is not None:
            try:
                cv2.imshow("soes_vision", vis)
                cv2.waitKey(1)
            except Exception as e:
                self.get_logger().warn(f"OpenCV imshow failed: {e}")

        # Estimasi diameter per cupcake (sangat sederhana)
        diam_mm = []
        for i in range(len(self.diam_mean)):
            if i < len(yolo_labels):
                _, cx, cy, bw, bh = yolo_labels[i]
                bw = max(1e-6, float(bw))
                # scaling dummy: pakai px_to_mm_ref sebagai referensi
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

    def destroy_node(self):
        # Pastikan kamera dan window rapi saat node mati
        if self.cap is not None:
            self.cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


# ===== helper =====

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
    """Fallback simple pakai threshold warna."""
    # BGR -> HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Range warna terang (cream) sangat kasar, silakan adjust
    lower = np.array([0, 0, 120], dtype=np.uint8)
    upper = np.array([180, 80, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter kontur kecil
    good = [c for c in cnts if cv2.contourArea(c) > 200]

    vis = draw_detected(img, good)
    yolo_labels = []  # tidak ada bbox dari YOLO di fallback

    return vis, good, yolo_labels


def detect_choux_from_frame(img):
    """Deteksi choux dari satu frame. Return (vis, good_cnts, yolo_labels)."""
    model = load_yolo_model()
    if model is None:
        # fallback tanpa YOLO
        return _detect_choux_color_fallback(img)

    # Coba panggil YOLO
    try:
        results = model(img, verbose=False)
    except TypeError:
        # beberapa versi pakai .predict
        results = model.predict(img, verbose=False)

    # Ambil result pertama
    if isinstance(results, (list, tuple)):
        res0 = results[0]
    else:
        res0 = results

    vis = img.copy()
    good = []
    yolo_labels = []

    # Ambil bbox dalam format xywh
    try:
        boxes_xywh = res0.boxes.xywh.cpu().numpy()
    except Exception:
        boxes_xywh = np.zeros((0, 4), dtype=float)

    for i, (cx, cy, w, h) in enumerate(boxes_xywh):
        cx_f, cy_f, w_f, h_f = float(cx), float(cy), float(w), float(h)
        yolo_labels.append((i, cx_f, cy_f, w_f, h_f))

        cx_i, cy_i = int(cx_f), int(cy_f)
        r = int(max(w_f, h_f) / 2.0)
        r = max(r, 5)

        cv2.circle(vis, (cx_i, cy_i), r, (0, 255, 0), 2)
        cv2.putText(vis, f"Choux {i + 1}", (cx_i - 40, cy_i - r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # bikin contour dummy satu titik supaya tipe datanya mirip
        cnt = np.array([[[cx_i, cy_i]]], dtype=np.int32)
        good.append(cnt)

    return vis, good, yolo_labels


def debug_detect_choux_from_usb(cam_index=0):
    """Debug langsung dari USB cam tanpa ROS."""
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[DEBUG] Failed to open camera index {cam_index}")
        return

    load_yolo_model()
    print("[DEBUG] Press ESC or q to quit debug window.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[DEBUG] Failed to read frame, stopping.")
            break

        vis, good, labels = detect_choux_from_frame(frame)
        cv2.imshow("soes_vision debug", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    rclpy.init()
    node = VisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
