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


class VisionNode(Node):
    def __init__(self):
        super().__init__('soes_vision')

        # parameters (centralized via soes_bringup/config/vision.yaml)
        self.declare_parameter('publish_rate_hz', 5.0)
        self.declare_parameter('frame_id', 'robot_base')
        self.declare_parameter('centers_m', [0.22, 0.05, 0.10,  0.20, 0.00, 0.10,  0.22, -0.05, 0.10])
        self.declare_parameter('diameter_mean_mm', [30.0, 30.0, 30.0])
        self.declare_parameter('quality_tolerance_mm', 3.0)

        self.rate = float(self.get_parameter('publish_rate_hz').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        arr = list(self.get_parameter('centers_m').value)
        self.centers = [(arr[0],arr[1],arr[2]), (arr[3],arr[4],arr[5]), (arr[6],arr[7],arr[8])]
        self.diam_mean = list(self.get_parameter('diameter_mean_mm').value)
        self.tol = float(self.get_parameter('quality_tolerance_mm').value)

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.centers_pub = self.create_publisher(CupcakeCenters, '/vision/centers', qos)
        self.quality_pub = self.create_publisher(VisionQuality, '/vision/quality', qos)

        self.k = 0
        self.timer = self.create_timer(max(0.001, 1.0/self.rate), self._on_timer)
        self.get_logger().info('soes_vision started (dummy publishers).')

    def _on_timer(self):
        now = self.get_clock().now().to_msg()

        # centers message
        msg_c = CupcakeCenters()
        msg_c.header.stamp = now
        msg_c.header.frame_id = self.frame_id
        msg_c.frame_id = self.frame_id
        for (x, y, z) in self.centers:
            p = Point(); p.x = float(x); p.y = float(y); p.z = float(z)
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
        msg_q.needs_human = (max(msg_q.diameter_mm) - min(msg_q.diameter_mm)) > self.tol
        self.quality_pub.publish(msg_q)

        self.k += 1


def main():
    rclpy.init()
    node = VisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

# -------- helpers --------
def draw_detected(img, cnts, color=(0, 255, 0)):
    out = img.copy()
    for i, c in enumerate(cnts, 1):
        (cx, cy), r = cv2.minEnclosingCircle(c)
        center, r = (int(cx), int(cy)), int(r)
        cv2.circle(out, center, r, color, 3)
        cv2.putText(out, f"Choux {i}", (center[0] - 40, center[1] - r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def detect_choux_from_frame(img):
    """
    Core detection pipeline adapted for a single frame (USB camera or still image).
    Returns:
        vis         : visualization image with circles & labels
        good_cnts   : list of contours considered as choux
        yolo_labels : list of [class_id, cx, cy, w, h] in YOLO normalized format
    """
    h, w = img.shape[:2]

    # 1) Color-based segmentation (HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # choux is yellowish with decent saturation & brightness; paper is low-S
    # Tune these 3 numbers if lighting changes a lot:
    S_min = 50
    V_min = 80
    mask_sv = (S > S_min) & (V > V_min)

    # Optionally restrict to yellow hues (helps if your background picks up S):
    # (uncomment to enforce hue)
    # mask_h = ((H >= 10) & (H <= 40))  # yellow range
    # mask = (mask_sv & mask_h).astype(np.uint8) * 255

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

    # Fill internal holes inside each blob
    # (distance transform + threshold is a robust fill)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, mask = cv2.threshold(dist, 3, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)

    # 3) Find candidate contours (ignore borders)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Relative size filters (adaptive to image size)
    img_area = h * w
    min_area = 0.003 * img_area   # ~0.3% of image
    max_area = 0.10 * img_area    # up to ~10% of image (tweak if needed)

    good = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue

        x, y, ww, hh = cv2.boundingRect(c)

        # Ignore contours touching the frame edges (background arcs/bleed)
        border_pad = 8
        if (
            x <= border_pad or y <= border_pad or
            (x + ww) >= (w - border_pad) or (y + hh) >= (h - border_pad)
        ):
            continue

        # Roundish check (kept loose because choux aren’t perfect disks)
        perim = cv2.arcLength(c, True)
        if perim == 0:
            continue
        circularity = 4 * np.pi * area / (perim * perim)  # 1.0 = perfect circle
        if circularity < 0.45:  # loosened threshold vs previous 0.7
            # try convex hull to heal swirl gaps
            hull = cv2.convexHull(c)
            area_h = cv2.contourArea(hull)
            perim_h = cv2.arcLength(hull, True)
            circ_h = 4 * np.pi * area_h / (perim_h * perim_h) if perim_h else 0
            if circ_h < 0.45:
                continue
            c = hull  # use healed shape

        # Aspect ratio near square-ish (but allow tilt)
        ar = ww / float(hh)
        if not (0.6 <= ar <= 1.6):
            continue

        good.append(c)

    # 4) Visualize + report
    vis = draw_detected(img, good, (0, 255, 0))

    # Matplotlib visualization (single frame; useful when testing from a notebook)
    plt.figure(figsize=(9, 12))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    print(f"Detected choux: {len(good)}")

    # Optional: export YOLO-style bboxes (class id 0) normalized to [0,1]
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
