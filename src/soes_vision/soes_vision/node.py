#!/usr/bin/env python3
import math
import time
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Point
from soes_msgs.msg import CupcakeCenters, VisionQuality
from sensor_msgs.msg import Image
from std_msgs.msg import Int32  # NEW: subscribe to /state/phase

import cv2
import numpy as np

# Try ultralytics
try:
    from ultralytics import YOLO as UltralyticsYOLO
    _HAS_ULTRALYTICS = True
except:
    UltralyticsYOLO = None
    _HAS_ULTRALYTICS = False

# Torchscript fallback
try:
    import torch
    _HAS_TORCH = True
except:
    torch = None
    _HAS_TORCH = False


class VisionNode(Node):
    def __init__(self):
        super().__init__('soes_vision')

        # Parameters
        self.declare_parameter('publish_rate_hz', 5.0)
        self.declare_parameter('camera_index', 0)

        self.rate = float(self.get_parameter('publish_rate_hz').value)
        self.camera_index = int(self.get_parameter('camera_index').value)

        # Force ultralytics model
        self.model_type = 'ultralytics'
        self.model_path = "/home/jetson/soes_ws/src/soes_vision/weights/best.pt"

        # Publishers
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.centers_pub = self.create_publisher(CupcakeCenters, '/vision/centers', qos)
        self.quality_pub = self.create_publisher(VisionQuality, '/vision/quality', qos)

        # NEW: subscribe to state phase
        self.state_phase = -1  # default unknown
        self.create_subscription(Int32, '/state/phase', self._on_phase, qos)

        # Camera
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.get_logger().error("Camera failed to open.")
        else:
            self.get_logger().info("Camera opened successfully.")

        # Load model
        self.model = None
        self.model_mode = 'none'
        self._load_model()

        # Timer
        self.timer = self.create_timer(1.0 / self.rate, self._on_timer)
        self.get_logger().info("Vision node started.")
        # YOLO FPS limiter
        self.last_yolo_time = 0
        self.yolo_interval = 0.20   # YOLO runs every 0.20 s → 5 FPS

    # ----------------------
    # Model Loading
    # ----------------------
    def _load_model(self):
        if _HAS_ULTRALYTICS:
            try:
                self.model = UltralyticsYOLO(self.model_path)
                self.model_mode = 'ultralytics'
                self.get_logger().info("Loaded YOLO model with Ultralytics.")
                return
            except Exception as e:
                self.get_logger().error(f"Ultralytics load error: {e}")

        self.get_logger().warn("No model loaded, using fallback.")
        self.model_mode = "color_fallback"

    # ----------------------
    # Phase callback
    # ----------------------
    def _on_phase(self, msg: Int32):
        self.state_phase = int(msg.data)

    # ----------------------
    # Timer callback
    # ----------------------
    def _on_timer(self):
        # Only run camera + YOLO when state machine is in CAMERA phase (Phase.CAMERA = 4)
        if self.state_phase != 4:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to capture frame.")
            return

        # LIVE VIEW (non-blocking & low CPU)
        cv2.imshow("Camera", frame)
        cv2.waitKey(1)

        # ---------------------------
        # YOLO RATE LIMITER (5 FPS)
        # ---------------------------
        now = time.time()
        run_yolo = False
        if now - self.last_yolo_time >= self.yolo_interval:
            self.last_yolo_time = now
            run_yolo = True

        if run_yolo and self.model_mode == "ultralytics":
            centers, diam, score = self._run_ultralytics(frame)
        else:
            centers, diam, score = [], [], []

        # Publish ROS messages
        msg_c = CupcakeCenters()
        msg_c.header.stamp = self.get_clock().now().to_msg()

        dummy_centers = [(0, 0, 0), (0.1, 0, 0), (0.2, 0, 0)]
        for (x, y, z) in dummy_centers:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = float(z)
            msg_c.centers.append(p)

        self.centers_pub.publish(msg_c)

        msg_q = VisionQuality()
        msg_q.header.stamp = msg_c.header.stamp
        msg_q.diameter_mm = diam if diam else [30.0, 30.0, 30.0]
        msg_q.score = score if score else [1.0, 1.0, 1.0]
        msg_q.needs_human = False

        self.quality_pub.publish(msg_q)

    # ----------------------
    # Ultralytics YOLO inference
    # ----------------------
    def _run_ultralytics(self, frame):
        try:
            results = self.model(frame, verbose=False)
            r = results[0]

            # SHOW YOLO OUTPUT
            yolo_img = r.plot()
            cv2.imshow("YOLO Output", yolo_img)
            cv2.waitKey(1)

            centers = []
            diams = []
            scores = []

            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf.cpu().numpy())
                scores.append(conf)

                cx = float((x1 + x2) / 2)
                cy = float((y1 + y2) / 2)
                centers.append((cx, cy))

                diam_px = (x2 - x1)
                diams.append(diam_px * 0.5)   # placeholder

            return centers, diams, scores

        except Exception as e:
            self.get_logger().error(f"YOLO error: {e}")
            return [], [], []


def main():
    rclpy.init()
    node = VisionNode()
    try:
        rclpy.spin(node)
    finally:
        node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
