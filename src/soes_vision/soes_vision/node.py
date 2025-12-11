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

_YOLO_MODEL = None  # cached model


def load_yolo_model(model_path: str = "/home/jetson/soes_ws/src/soes_vision/weights/best.pt"):
    """Load YOLO model once, cache globally."""
    global _YOLO_MODEL
    if not _HAS_YOLO:
        return None

    if _YOLO_MODEL is not None:
        return _YOLO_MODEL

    try:
        _YOLO_MODEL = UltralyticsYOLO(model_path)
    except Exception:
        _YOLO_MODEL = None

    return _YOLO_MODEL


class VisionNode(Node):
    def __init__(self):
        super().__init__('soes_vision')

        # -------- parameters --------
        self.declare_parameter('publish_rate_hz', 5.0)
        self.declare_parameter('frame_id', 'robot_base')
        self.declare_parameter('centers_m', [
            0.22, 0.05, 0.10,
            0.20, 0.00, 0.10,
            0.22, -0.05, 0.10
        ])
        self.declare_parameter('diameter_mean_mm', [30.0, 30.0, 30.0])
        self.declare_parameter('quality_tolerance_mm', 3.0)
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('px_to_mm_ref', 0.1)
        self.declare_parameter('visualize', False)

        # extract
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

        # flag for visualization overlay
        self.camera_phase = False

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        # publishers
        self.centers_pub = self.create_publisher(CupcakeCenters, '/vision/centers', qos)
        self.quality_pub = self.create_publisher(VisionQuality, '/vision/quality', qos)
        self.soess_done_pub = self.create_publisher(Bool, '/vision/soes_done', qos)

        # Subscriber (Vision Code)
        self.request_sub = self.create_subscription(
            Bool,
            '/vision/request',
            self._on_request,
            vision_qos
        )

        # open camera
        self.cap = cv2.VideoCapture(self.cam_index)
        if not self.cap.isOpened():
            self.get_logger().error(f"Camera index {self.cam_index} failed to open!")
            self.cap = None

        # load YOLO once
        load_yolo_model()

        # main timer
        self.timer = self.create_timer(max(0.001, 1.0 / self.rate), self._on_timer)

        # visualization timer
        self.vis_timer = None
        if self.visualize:
            self.vis_timer = self.create_timer(0.1, self._on_vis_timer)

        self.get_logger().info('soes_vision started.')

    # -------------- periodic centers publisher --------------
    def _on_timer(self):
        now = self.get_clock().now().to_msg()

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

    # -------------- visualization loop --------------
    def _on_vis_timer(self):
        if not self.visualize:
            return
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return

        try:
            vis, good_cnts, yolo_labels = detect_choux_from_frame(frame)
        except Exception as e:
            self.get_logger().warn(f"detect_choux_from_frame failed in vis_timer: {e}")
            vis = frame.copy()

        text = f"CAMERA Phase = {self.camera_phase}"
        cv2.putText(vis, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        try:
            cv2.imshow("soes_vision", vis)
            cv2.waitKey(1)
        except Exception:
            pass

    # -------------- request handler → detect once --------------
    def _on_request(self, msg: Bool):
        self.camera_phase = True

        model = load_yolo_model()
        if self.cap is None or not self.cap.isOpened():
            self.get_logger().error("Camera not opened.")
            self.camera_phase = False
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.get_logger().error("Failed to read frame.")
            self.camera_phase = False
            return

        vis, good_cnts, yolo_labels = detect_choux_from_frame(frame)

        # no choux detected
        no_choux = (len(good_cnts) == 0 and len(yolo_labels) == 0)
        if no_choux:
            self.get_logger().error("VISION request: no choux detected.")

        # diameter estimation
        diam_mm = []
        for i in range(len(self.diam_mean)):
            if i < len(yolo_labels):
                _, cx, cy, bw, bh = yolo_labels[i]
                bw = max(1e-6, float(bw))
                est = float(self.diam_mean[i]) * (bw / max(1e-6, self.px_to_mm_ref))
                diam_mm.append(est)
            else:
                diam_mm.append(float(self.diam_mean[i]))

        msg_q = VisionQuality()
        msg_q.header.stamp = self.get_clock().now().to_msg()
        msg_q.diameter_mm = [float(x) for x in diam_mm]
        msg_q.score = [1.0] * len(diam_mm)

        if no_choux:
            msg_q.needs_human = True
        else:
            msg_q.needs_human = (max(msg_q.diameter_mm) - min(msg_q.diameter_mm)) > self.tol

        self.quality_pub.publish(msg_q)

        soes_done_msg = Bool()
        soes_done_msg.data = (not msg_q.needs_human)
        self.soess_done_pub.publish(soes_done_msg)

        self.camera_phase = False

    # -------------- cleanup --------------
    def destroy_node(self):
        if self.cap is not None:
            self.cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


# ===== helper detection functions =====

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
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 0, 120], dtype=np.uint8)
    upper = np.array([180, 80, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    good = [c for c in cnts if cv2.contourArea(c) > 200]
    vis = draw_detected(img, good)
    return vis, good, []


def detect_choux_from_frame(img):
    model = load_yolo_model()
    if model is None:
        return _detect_choux_color_fallback(img)

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

        cnt = np.array([[[cx_i, cy_i]]], dtype=np.int32)
        good.append(cnt)

    return vis, good, yolo_labels


def main():
    rclpy.init()
    node = VisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
