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
        # Reuse cached model
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

        self.get_logger().info("Initializing VisionNode...step: declare parameters")
        # -------- parameters --------
        self.declare_parameter('publish_rate_hz', 5.0)
        self.declare_parameter('frame_id', 'robot_base')
        self.declare_parameter(
            'centers_m',
            [0.22, 0.05, 0.10,
             0.20, 0.00, 0.10,
             0.22, -0.05, 0.10]
        )
        self.declare_parameter('diameter_mean_mm', [30.0, 30.0, 30.0])
        self.declare_parameter('quality_tolerance_mm', 3.0)
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('px_to_mm_ref', 0.1)
        self.declare_parameter('visualize', False)

        self.get_logger().info("Step: extract parameters")
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

        self.get_logger().info("Step: complete parameter extraction")

        # flag untuk tulisan CAMERA Phase = True/False
        self.camera_phase = False

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )
        self.get_logger().info("Step: create publishers")
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

        self.get_logger().info("Step: create subscription")
        # -------- subscriber --------
        self.request_sub = self.create_subscription(
            Bool, '/vision/request', self._on_request, qos
        )

        self.get_logger().info("Step: open camera ONCE at startup")
        # ---------- Kamera: buka sekali ----------
        self.cap = cv2.VideoCapture(self.cam_index)
        if not self.cap.isOpened():
            self.get_logger().error(
                f"Camera index {self.cam_index} failed to open!"
            )
            self.cap = None
        else:
            self.get_logger().info("Camera opened ONCE at startup.")

        self.get_logger().info("YOLO: starting initial model load...")
        load_yolo_model()
        self.get_logger().info("YOLO: initial model load done.")

        self.k = 0

        self.get_logger().info("Step: create main timer")
        # Timer hanya untuk publish CENTERS / info statis
        self.timer = self.create_timer(
            max(0.001, 1.0 / self.rate),
            self._on_timer
        )

        self.vis_timer = None
        if self.visualize:
            self.get_logger().info("Step: create visualization timer")
            # 10 Hz visualisasi + YOLO
            self.vis_timer = self.create_timer(0.1, self._on_vis_timer)

        self.get_logger().info('soes_vision started (init done).')

    def _on_timer(self):
        self.get_logger().debug("Entering _on_timer")
        now = self.get_clock().now().to_msg()
        self.get_logger().debug(f"Current time: {now}")

        # publish centers (saat ini masih statis)
        msg_c = CupcakeCenters()
        msg_c.header.stamp = now
        msg_c.header.frame_id = self.frame_id
        msg_c.frame_id = self.frame_id

        self.get_logger().debug("Appending centers to msg_c")
        for (x, y, z) in self.centers:
            self.get_logger().debug(f"Center: x={x}, y={y}, z={z}")
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = float(z)
            msg_c.centers.append(p)

        self.get_logger().debug("Publishing centers")
        self.centers_pub.publish(msg_c)

        self.k += 1
        self.get_logger().debug(f"Timer incremented k={self.k}")

    def _on_vis_timer(self):
        """Live camera view sejak start, dengan YOLO untuk overlay visual."""
        self.get_logger().debug("Entering _on_vis_timer")
        if not self.visualize:
            self.get_logger().debug("Visualize parameter is False, skipping visualization")
            return
        if self.cap is None or not self.cap.isOpened():
            self.get_logger().debug("Camera is None or not opened, skipping visualization")
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.get_logger().warn("Failed to read frame from camera for visualization.")
            return

        # Jalankan YOLO / fallback untuk visualisasi dari awal
        try:
            self.get_logger().debug("Running detect_choux_from_frame for visualization")
            vis, good_cnts, yolo_labels = detect_choux_from_frame(frame)
        except Exception as e:
            self.get_logger().warn(f"detect_choux_from_frame failed in vis_timer: {e}")
            vis = frame.copy()

        # Overlay text CAMERA phase di hasil deteksi
        self.get_logger().debug("Overlaying CAMERA phase text on frame")
        text = f"CAMERA Phase = {self.camera_phase}"
        cv2.putText(
            vis,
            text,
            (10, 30),  # kiri atas
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        try:
            self.get_logger().debug("Showing frame in OpenCV window: soes_vision")
            cv2.imshow("soes_vision", vis)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().warn(f"OpenCV imshow failed: {e}")

    def _on_request(self, msg: Bool):
        self.get_logger().info("VISION REQUEST received, running YOLO on one frame...")
        self.camera_phase = True
        self.get_logger().debug("Camera phase set to True.")

        model = load_yolo_model()
        if model is None:
            self.get_logger().warn(
                "YOLO model not available, using color-based fallback."
            )
        else:
            self.get_logger().debug("YOLO model loaded.")

        if self.cap is None or not self.cap.isOpened():
            self.get_logger().error("Camera not opened, cannot capture frame.")
            self.camera_phase = False
            return

        self.get_logger().debug("Reading frame from camera for request")
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.get_logger().error("Failed to read frame from camera.")
            self.camera_phase = False
            return

        self.get_logger().info("Running detection (YOLO or fallback)")
        vis, good_cnts, yolo_labels = detect_choux_from_frame(frame)

        # ===== ADDED: handle case when no choux detected =====
        no_choux = (len(good_cnts) == 0 and len(yolo_labels) == 0)
        if no_choux:
            self.get_logger().error(
                "VISION (on-request): no choux detected in frame. Human inspection required."
            )
        # =====================================================

        diam_mm = []
        for i in range(len(self.diam_mean)):
            if i < len(yolo_labels):
                _, cx, cy, bw, bh = yolo_labels[i]
                bw = max(1e-6, float(bw))
                est = float(self.diam_mean[i]) * (bw / max(1e-6, self.px_to_mm_ref))
                self.get_logger().debug(
                    f"YOLO label idx={i}: cx={cx}, cy={cy}, bw={bw}, estimated diameter={est}"
                )
                diam_mm.append(est)
            else:
                diam_mm.append(float(self.diam_mean[i]))
                self.get_logger().debug(
                    f"No YOLO label for idx={i}, using mean diameter={self.diam_mean[i]}"
                )

        self.get_logger().debug(f"Final diam_mm list: {diam_mm}")

        msg_q = VisionQuality()
        msg_q.header.stamp = self.get_clock().now().to_msg()
        msg_q.diameter_mm = [float(x) for x in diam_mm]
        msg_q.score = [1.0] * len(diam_mm)

        # ===== MODIFIED: force needs_human True when no choux =====
        if no_choux:
            msg_q.needs_human = True
        else:
            msg_q.needs_human = (
                max(msg_q.diameter_mm) - min(msg_q.diameter_mm)
            ) > self.tol
        # =========================================================

        self.get_logger().info(
            f"Publishing VisionQuality: diam_mm={msg_q.diameter_mm}, needs_human={msg_q.needs_human}"
        )
        self.quality_pub.publish(msg_q)

        soes_done_msg = Bool()
        soes_done_msg.data = (not msg_q.needs_human)
        self.get_logger().info(f"Publishing soes_done={soes_done_msg.data}")
        self.soess_done_pub.publish(soes_done_msg)

        if msg_q.needs_human:
            self.get_logger().warn('VISION (on-request): needs_human == True')
        else:
            self.get_logger().info('VISION (on-request): needs_human == False')

        self.camera_phase = False
        self.get_logger().debug("Camera phase set to False (request done)")

    def destroy_node(self):
        self.get_logger().info("Destroying VisionNode, cleaning up resources...")
        if self.cap is not None:
            self.get_logger().debug("Releasing camera resource.")
            self.cap.release()
        try:
            self.get_logger().debug("Destroying all OpenCV windows.")
            cv2.destroyAllWindows()
        except Exception:
            self.get_logger().warn("Exception during OpenCV window cleanup.")
        super().destroy_node()
        self.get_logger().info("VisionNode destroyed.")

# ===== helper =====

def draw_detected(img, cnts, color=(0, 255, 0)):
    out = img.copy()
    print("[draw_detected] Drawing contours...")
    for i, c in enumerate(cnts, 1):
        (cx, cy), r = cv2.minEnclosingCircle(c)
        center, r = (int(cx), int(cy)), int(r)
        cv2.circle(out, center, r, color, 3)
        cv2.putText(out, f"Choux {i}", (center[0] - 40, center[1] - r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    print(f"[draw_detected] Done drawing {len(cnts)} contours.")
    return out

def _detect_choux_color_fallback(img):
    """Fallback simple pakai threshold warna."""
    print("[_detect_choux_color_fallback] BGR -> HSV conversion")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    print("[_detect_choux_color_fallback] Creating mask by color range")
    lower = np.array([0, 0, 120], dtype=np.uint8)
    upper = np.array([180, 80, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    print("[_detect_choux_color_fallback] Blurring mask")
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"[_detect_choux_color_fallback] Found {len(cnts)} contours. Filtering...")
    good = [c for c in cnts if cv2.contourArea(c) > 200]

    print(f"[_detect_choux_color_fallback] {len(good)} contours after filtering. Drawing...")
    vis = draw_detected(img, good)
    yolo_labels = []

    print("[_detect_choux_color_fallback] Return visualized image and labels.")
    return vis, good, yolo_labels

def detect_choux_from_frame(img):
    """Deteksi choux dari satu frame. Return (vis, good_cnts, yolo_labels)."""
    print("[detect_choux_from_frame] Loading YOLO model (if available)...")
    model = load_yolo_model()
    if model is None:
        print("[detect_choux_from_frame] YOLO unavailable, using color fallback.")
        return _detect_choux_color_fallback(img)

    print("[detect_choux_from_frame] Running YOLO model...")
    try:
        results = model(img, verbose=False)
    except TypeError:
        results = model.predict(img, verbose=False)

    if isinstance(results, (list, tuple)):
        res0 = results[0]
    else:
        res0 = results

    vis = img.copy()
    good = []
    yolo_labels = []

    try:
        boxes_xywh = res0.boxes.xywh.cpu().numpy()
        print(f"[detect_choux_from_frame] Detected {len(boxes_xywh)} boxes from YOLO.")
    except Exception:
        boxes_xywh = np.zeros((0, 4), dtype=float)
        print("[detect_choux_from_frame] Could not extract box info from YOLO result.")

    for i, (cx, cy, w, h) in enumerate(boxes_xywh):
        print(f"[detect_choux_from_frame] Drawing Box#{i+1}: center=({cx},{cy}), w={w}, h={h}")
        cx_f, cy_f, w_f, h_f = float(cx), float(cy), float(w), float(h)
        yolo_labels.append((i, cx_f, cy_f, w_f, h_f))

        cx_i, cy_i = int(cx_f), int(cy_f)
        r = int(max(w_f, h_f) / 2.0)
        r = max(r, 5)

        cv2.circle(vis, (cx_i, cy_i), r, (0, 255, 0), 2)
        cv2.putText(vis, f"Choux {i + 1}", (cx_i - 40, cy_i - r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cnt = np.array([[[cx_i, cy_i]]], dtype=np.int32)
        good.append(cnt)

    print("[detect_choux_from_frame] Done. Returning visual, good, yolo_labels.")
    return vis, good, yolo_labels

def debug_detect_choux_from_usb(cam_index=0):
    """Debug langsung dari USB cam tanpa ROS."""
    print("[debug_detect_choux_from_usb] Opening camera for debug...")
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

        print("[debug_detect_choux_from_usb] Detecting choux from frame...")
        vis, good, labels = detect_choux_from_frame(frame)
        cv2.imshow("soes_vision debug", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            print("[debug_detect_choux_from_usb] Quit key pressed, exiting.")
            break

    print("[debug_detect_choux_from_usb] Releasing camera and destroying windows.")
    cap.release()
    cv2.destroyAllWindows()


def main():
    print("[main] Initializing rclpy...")
    rclpy.init()
    print("[main] Creating VisionNode...")
    node = VisionNode()
    print("[main] Spinning node...")
    rclpy.spin(node)
    print("[main] Destroying node and shutting down rclpy...")
    node.destroy_node()
    rclpy.shutdown()
    print("[main] Done.")

if __name__ == "__main__":
    main()
