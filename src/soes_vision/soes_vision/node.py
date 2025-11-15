import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Point
from soes_msgs.msg import CupcakeCenters, VisionQuality

import cv2
import numpy as np


class VisionNode(Node):
    def __init__(self):
        super().__init__('soes_vision')

        # parameters
        self.declare_parameter('publish_rate_hz', 5.0)
        self.declare_parameter('frame_id', 'robot_base')
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('visualize', False)

        self.rate = float(self.get_parameter('publish_rate_hz').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.cam_index = int(self.get_parameter('camera_index').value)
        self.visualize = bool(self.get_parameter('visualize').value)

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        self.pub_centers = self.create_publisher(CupcakeCenters, '/vision/centers', qos)
        self.pub_quality = self.create_publisher(VisionQuality, '/vision/quality', qos)

        # ---- open camera ----
        self.cap = cv2.VideoCapture(self.cam_index)
        if not self.cap.isOpened():
            self.get_logger().error("failed to open camera index")
        else:
            self.get_logger().info("VisionNode camera opened at index")

        self.timer = self.create_timer(1.0 / self.rate, self._on_timer)
        self.get_logger().info("SOES VisionNode started.")

    def _on_timer(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning("VisionNode: camera frame read failed")
            return

        vis, contours, yolo = detect_choux_from_frame(frame)

        # --- publish centers ---
        msg_c = CupcakeCenters()
        msg_c.header.stamp = self.get_clock().now().to_msg()
        msg_c.header.frame_id = self.frame_id
        msg_c.frame_id = self.frame_id

        for c in contours:
            (cx, cy), r = cv2.minEnclosingCircle(c)
            P = Point()
            P.x = float(cx)     # these are pixel coords now
            P.y = float(cy)
            P.z = float(r)      # repurpose z = pixel radius for now
            msg_c.centers.append(P)

        self.pub_centers.publish(msg_c)

        # compute cupcake diameter in mm (placeholder scaling)
        pixel_to_mm = 0.25 #dummy mapping pixel_to_mm// calibrate later
        diameters = []
        for c in contours:
            (_, _), r = cv2.minEnclosingCircle(c)
            diameters.append(float(2 * r * pixel_to_mm))

        msg_q = VisionQuality()
        msg_q.header.stamp = msg_c.header.stamp
        msg_q.diameter_mm = diameters
        msg_q.score = [1.0] * len(diameters)
        msg_q.needs_human = (len(diameters) > 0 and max(diameters)-min(diameters) > 3.0)

        self.pub_quality.publish(msg_q)

        # optional visualization
        if self.visualize:
            cv2.imshow("SOES Vision", vis)
            cv2.waitKey(1)

    def destroy_node(self):
        if hasattr(self, "cap"):
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


# DETECTION PIPELINE (unchanged except removing matplotlib)
def draw_detected(img, cnts, color=(0, 255, 0)):
    out = img.copy()
    for i, c in enumerate(cnts, 1):
        (cx, cy), r = cv2.minEnclosingCircle(c)
        center = (int(cx), int(cy))
        r = int(r)
        cv2.circle(out, center, r, color, 3)
        cv2.putText(out, f"C{i}", (center[0] - 30, center[1] - r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def detect_choux_from_frame(img):
    h, w = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    S_min = 50
    V_min = 80
    mask = ((S > S_min) & (V > V_min)).astype(np.uint8) * 255

    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, mask = cv2.threshold(dist, 3, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = h * w
    min_area = 0.003 * img_area
    max_area = 0.10 * img_area

    good = []
    for c in cnts:
        area = cv2.contourArea(c)
        if not (min_area < area < max_area):
            continue

        (x, y, ww, hh) = cv2.boundingRect(c)
        border = 8
        if x <= border or y <= border or x + ww >= w - border or y + hh >= h - border:
            continue

        good.append(c)

    vis = draw_detected(img, good)
    return vis, good, None


def main():
    rclpy.init()
    node = VisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
