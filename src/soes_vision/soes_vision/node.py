#!/usr/bin/env python3 
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Point
from soes_msgs.msg import CupcakeCenters, VisionQuality

# ===== Added imports for vision pipeline =====
import cv2
import numpy as np
import matplotlib.pyplot as plt
from std_msgs.msg import Bool

# ===== NEW: YOLO (Ultralytics) support =====
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
    """
    Lazy-load YOLO model once, reuse for all frames.
    If YOLO not installed or load fails, returns None.
    """
    global _YOLO_MODEL
    if not _HAS_YOLO:
        print("[YOLO] Ultralytics not available, using color-based fallback.")
        return None

    if _YOLO_MODEL is not None:
        return _YOLO_MODEL

    try:
        _YOLO_MODEL = UltralyticsYOLO(model_path)
        print(f"[YOLO] Loaded model from: {model_path}")
    except Exception as e:
        print(f"[YOLO] Failed to load model: {e}")
        _YOLO_MODEL = None

    return _YOLO_MODEL


class VisionNode(Node):
    def __init__(self):
        super().__init__('soes_vision')

        # parameters (centralized via soes_bringup/config/vision.yaml)
        self.declare_parameter('publish_rate_hz', 5.0)
        self.declare_parameter('frame_id', 'robot_base')
        self.declare_parameter('centers_m', [0.22, 0.05, 0.10,
                                             0.20, 0.00, 0.10,
                                             0.22, -0.05, 0.10])
        self.declare_parameter('diameter_mean_mm', [30.0, 30.0, 30.0])
        self.declare_parameter('quality_tolerance_mm', 3.0)

        # --- new: camera index and px->mm reference (optional tuning) ---
        # px_to_mm is only used as a rough conversion to estimate diameters
        # from normalized YOLO bbox widths — it's conservative and intended
        # for making a reasonable `needs_human` estimate on a single frame.
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('px_to_mm_ref', 0.1)  # reference normalized width

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

        # new vision config
        self.cam_index = int(self.get_parameter('camera_index').value)
        self.px_to_mm_ref = float(self.get_parameter('px_to_mm_ref').value)

        qos = QoSProfile(depth=10,
                         reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)
        self.centers_pub = self.create_publisher(CupcakeCenters,
                                                 '/vision/centers', qos)
        self.quality_pub = self.create_publisher(VisionQuality,
                                                 '/vision/quality', qos)

        # ---------- NEW: publisher for the SOES decision and request handling ----------
        # Publish final decision for state machine / I2C bridge:
        # Topic: /vision/soes_done (Bool) -> True == done, False == needs human
        self.soess_done_pub = self.create_publisher(Bool, '/vision/soes_done', qos)

        # Subscribe to requests from state machine: when a /vision/request Bool(True)
        # is received we will run a single-frame detection and publish the decision.
        self.request_sub = self.create_subscription(Bool, '/vision/request',
                                                    self._on_request, qos)

        self.k = 0
        self.timer = self.create_timer(max(0.001, 1.0/self.rate),
                                       self._on_timer)
        self.get_logger().info('soes_vision started (dummy publishers).')

    def _on_timer(self):
        now = self.get_clock().now().to_msg()

        # centers message
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

        # “quality” message (synthetic wiggle)
        d0 = self.diam_mean[0] + 1.0 * math.sin(self.k * 0.20)
        d1 = self.diam_mean[1] + 0.5 * math.sin(self.k * 0.15 + 1.0)
        d2 = self.diam_mean[2] + 0.7 * math.sin(self.k * 0.18 + 2.0)

        msg_q = VisionQuality()
        msg_q.header.stamp = now
        msg_q.diameter_mm = [float(d0), float(d1), float(d2)]
        msg_q.score = [1.0, 1.0, 1.0]
        msg_q.needs_human = (
            max(msg_q.diameter_mm) - min(msg_q.diameter_mm)
        ) > self.tol
        self.quality_pub.publish(msg_q)

        # Note: we DO NOT automatically publish /vision/soes_done here; the
        # state machine should request a fresh reading via /vision/request.
        # However for backwards compatibility we also publish the most-recent
        # synthetic decision in case other nodes rely on it.
        soes_done_msg = Bool()
        soes_done_msg.data = (not msg_q.needs_human)
        self.soess_done_pub.publish(soes_done_msg)

        self.k += 1

    # ---------- NEW: Request handler (run detection once when asked) ----------
    def _on_request(self, msg: Bool):
        """
        When a request arrives (Bool; content ignored, treat as "run now"),
        capture a single frame from the configured camera and run detection.
        Publish VisionQuality and /vision/soes_done based on VisionQuality.needs_human.
        """
        # Load the YOLO model early if available (lazy load)
        load_yolo_model()

        # Capture a single frame from camera index (best-effort)
        cap = cv2.VideoCapture(self.cam_index)
        if not cap.isOpened():
            self.get_logger().warn(f'Camera index {self.cam_index} open failed. Using last synthetic quality.')
            # If camera unavailable, publish the most recent synthetic quality we already produce
            # (this keeps the state machine robust to camera failures)
            # Note: The synthetic `quality` from the timer was already published
            return

        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            self.get_logger().warn('Failed to capture frame on request. Using last synthetic quality.')
            return

        # Run detection on this single frame
        vis, good_cnts, yolo_labels = detect_choux_from_frame(frame)

        # Build a VisionQuality message derived from detections.
        # We estimate diameters (mm) by mapping normalized bbox width to mean diameter.
        # This is a heuristic: diameter_mm_i = diam_mean_i * (bw / px_to_mm_ref)
        # where bw is normalized bbox width from YOLO (0..1).
        diam_mm = []
        for i in range(len(self.diam_mean)):
            if i < len(yolo_labels):
                _, cx, cy, bw, bh = yolo_labels[i]
                # avoid zero
                bw = max(1e-6, float(bw))
                est = float(self.diam_mean[i]) * (bw / max(1e-6, self.px_to_mm_ref))
                diam_mm.append(est)
            else:
                # no detection for this slot -> fall back to nominal mean
                diam_mm.append(float(self.diam_mean[i]))

        msg_q = VisionQuality()
        msg_q.header.stamp = self.get_clock().now().to_msg()
        msg_q.diameter_mm = [float(x) for x in diam_mm]
        msg_q.score = [1.0] * len(diam_mm)
        msg_q.needs_human = (max(msg_q.diameter_mm) - min(msg_q.diameter_mm)) > self.tol

        # Publish VisionQuality as before
        self.quality_pub.publish(msg_q)

        # Publish /vision/soes_done (True==done, False==needs human)
        soes_done_msg = Bool()
        soes_done_msg.data = (not msg_q.needs_human)
        self.soess_done_pub.publish(soes_done_msg)

        # Debug logging
        if msg_q.needs_human:
            self.get_logger().warn('VISION (on-request): needs_human == True -> reporting SOES_NOT_DONE')
        else:
            self.get_logger().info('VISION (on-request): needs_human == False -> reporting SOES_DONE')

    # rest of file unchanged (helpers and detection pipeline) ...

# -------- helpers --------
def draw_detected(img, cnts, color=(0, 255, 0)):
    """
    Unchanged helper: expects contours.
    For YOLO, we synthesize simple rectangular contours from bbox.
    """
    out = img.copy()
    for i, c in enumerate(cnts, 1):
        (cx, cy), r = cv2.minEnclosingCircle(c)
        center, r = (int(cx), int(cy)), int(r)
        cv2.circle(out, center, r, color, 3)
        cv2.putText(out, f"Choux {i}", (center[0] - 40, center[1] - r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def _detect_choux_color_fallback(img):
    """
    ORIGINAL HSV + contour pipeline, kept as fallback when YOLO
    is not available or fails.
    """
    h, w = img.shape[:2]

    # 1) Color-based segmentation (HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    S_min = 50
    V_min = 80
    mask_sv = (S > S_min) & (V > V_min)
    mask = mask_sv.astype(np.uint8) * 255

    # 2) Clean mask: close gaps, open small noise, fill holes
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    )

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, mask = cv2.threshold(dist, 3, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

    img_area = h * w
    min_area = 0.003 * img_area
    max_area = 0.10 * img_area

    good = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        x, y, ww, hh = cv2.boundingRect(c)
        border_pad = 8
        if (
            x <= border_pad or y <= border_pad or
            (x + ww) >= (w - border_pad) or (y + hh) >= (h - border_pad)
        ):
            continue

        perim = cv2.arcLength(c, True)
        if perim == 0:
            continue
        circularity = 4 * np.pi * area / (perim * perim)
        if circularity < 0.45:
            hull = cv2.convexHull(c)
            area_h = cv2.contourArea(hull)
            perim_h = cv2.arcLength(hull, True)
            circ_h = 4 * np.pi * area_h / (perim_h * perim_h) if perim_h else 0
            if circ_h < 0.45:
                continue
            c = hull

        ar = ww / float(hh)
        if not (0.6 <= ar <= 1.6):
            continue

        good.append(c)

    vis = draw_detected(img, good, (0, 255, 0))

    plt.figure(figsize=(9, 12))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    print(f"Detected choux (color fallback): {len(good)}")

    yolo_labels = []
    for c in good:
        x, y, ww, hh = cv2.boundingRect(c)
        cx, cy = x + ww / 2, y + hh / 2
        cls = 0
        yolo_labels.append([cls, cx / w, cy / h, ww / w, hh / h])

    for lab in yolo_labels:
        cls, cx, cy, bw, bh = lab
        print(f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return vis, good, yolo_labels


def detect_choux_from_frame(img):
    """
    Core detection pipeline adapted for a single frame (USB camera or still image).

    NOW:
      - Try YOLO (Ultralytics)
      - If not available, fall back to the original HSV+contour logic.

    Returns:
        vis         : visualization image with circles & labels
        good_cnts   : list of contours considered as choux
        yolo_labels : list of [class_id, cx, cy, w, h] in YOLO normalized format
    """
    h, w = img.shape[:2]

    # Try YOLO first
    model = load_yolo_model()
    if model is None:
        # Fallback: original OpenCV pipeline
        return _detect_choux_color_fallback(img)

    try:
        results = model(img, verbose=False)
        r = results[0]

        good = []
        yolo_labels = []

        # If multiple classes exist, you can filter with box.cls
        for box in r.boxes:
            # Bounding box in absolute pixels
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

            # Confidence & class (if needed)
            conf = float(box.conf.cpu().numpy())
            cls_id = int(box.cls.cpu().numpy()) if hasattr(box, "cls") else 0

            # Normalized YOLO label
            bw = x2 - x1
            bh = y2 - y1
            cx = x1 + bw / 2.0
            cy = y1 + bh / 2.0

            # Store YOLO-style label (normalized)
            yolo_labels.append([cls_id, cx / w, cy / h, bw / w, bh / h])

            # Synthesize a simple rectangular contour from bbox
            cnt = np.array(
                [
                    [x1, y1],
                    [x2, y1],
                    [x2, y2],
                    [x1, y2],
                ],
                dtype=np.int32,
            ).reshape((-1, 1, 2))
            good.append(cnt)

        vis = draw_detected(img, good, (0, 255, 0))

        # Optional YOLO debug visualization (comment out if not needed)
        plt.figure(figsize=(9, 12))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(f"YOLO detections: {len(good)}")
        plt.show()

        print(f"Detected choux (YOLO): {len(good)}")
        for lab in yolo_labels:
            cls, cx, cy, bw, bh = lab
            print(f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        return vis, good, yolo_labels

    except Exception as e:
        print(f"[YOLO] Error during inference, using color fallback: {e}")
        return _detect_choux_color_fallback(img)


def debug_detect_choux_from_usb(cam_index=0):
    """
    Helper to use your USB camera on Jetson Nano.
    Call this manually (NOT used by ROS2 entry point), e.g. from a separate script
    or Python shell, to monitor the top-right camera and see detected choux.
    """
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"Failed to open camera index {cam_index}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera")
            break

        vis, good, yolo_labels = detect_choux_from_frame(frame)

        # For live monitoring with OpenCV window
        cv2.imshow("choux_detect", vis)

        # ESC to quit
        if cv2.waitKey(1) & 0xFF == 27:
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
