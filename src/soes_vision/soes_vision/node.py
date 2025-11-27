#!/usr/bin/env python3
import math
import time
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Point
from soes_msgs.msg import CupcakeCenters, VisionQuality

# vision imports
import cv2
import numpy as np

# try to import ultralytics YOLO class (common on Jetson if you installed ultralytics)
try:
    from ultralytics import YOLO as UltralyticsYOLO
    _HAS_ULTRALYTICS = True
except Exception:
    UltralyticsYOLO = None
    _HAS_ULTRALYTICS = False

# torch fallback for torchscript models
try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False


class VisionNode(Node):
    def __init__(self):
        super().__init__('soes_vision')

        # -------------------
        # ROS parameters (can be set in YAML / launch)
        # -------------------
        self.declare_parameter('publish_rate_hz', 5.0)                 # inference / publish rate
        self.declare_parameter('frame_id', 'robot_base')              # TF frame for published centers
        self.declare_parameter('camera_index', 0)                     # index for cv2.VideoCapture
        self.declare_parameter('model_path', 'yolov11.pt')            # path to your model on the Jetson workspace
        self.declare_parameter('model_type', 'auto')                 # 'auto'|'ultralytics'|'torchscript'|'color_fallback'
        self.declare_parameter('centers_m', [0.22, 0.05, 0.10,  0.20, 0.00, 0.10,  0.22, -0.05, 0.10])
        self.declare_parameter('diameter_mean_mm', [30.0, 30.0, 30.0])
        self.declare_parameter('quality_tolerance_mm', 3.0)

        # parameter values
        self.rate = float(self.get_parameter('publish_rate_hz').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.camera_index = int(self.get_parameter('camera_index').value)
        self.model_path = str(self.get_parameter('model_path').value)
        self.model_type = str(self.get_parameter('model_type').value).lower()

        arr = list(self.get_parameter('centers_m').value)
        self.centers = [(arr[0],arr[1],arr[2]), (arr[3],arr[4],arr[5]), (arr[6],arr[7],arr[8])]
        self.diam_mean = list(self.get_parameter('diameter_mean_mm').value)
        self.tol = float(self.get_parameter('quality_tolerance_mm').value)

        # ROS publishers
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        self.centers_pub = self.create_publisher(CupcakeCenters, '/vision/centers', qos)
        self.quality_pub = self.create_publisher(VisionQuality, '/vision/quality', qos)

        # camera capture (OpenCV)
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open camera index {self.camera_index}. Node will continue but no frames will be processed.")
        else:
            self.get_logger().info(f"Opened camera index {self.camera_index} for capture.")

        # load model (attempt multiple strategies)
        self.model = None
        self.model_mode = 'none'
        self._load_model()

        # timer
        self._t_last = time.time()
        self.k = 0
        self.timer = self.create_timer(max(0.001, 1.0/self.rate), self._on_timer)
        self.get_logger().info('soes_vision node started.')

    # -------------------
    # Model loading
    # -------------------
    def _load_model(self):
        """
        Attempt to load the user's model using a small priority:
         1) explicit 'ultralytics' if requested AND package present
         2) explicit 'torchscript' if requested and torch available
         3) 'auto' tries ultralytics then torchscript
         4) if nothing works, fall back to color-based detector (in-node)
        """
        mp = self.model_path
        requested = self.model_type
        self.get_logger().info(f"Loading model path='{mp}' requested_type='{requested}'.")

        # Helper: try ultralytics
        def try_ultralytics():
            if not _HAS_ULTRALYTICS:
                return False
            try:
                m = UltralyticsYOLO(mp)
                # simple single inference warm-up (no camera required)
                self.model = m
                self.model_mode = 'ultralytics'
                self.get_logger().info("Loaded model with ultralytics.YOLO")
                return True
            except Exception as e:
                self.get_logger().warn(f"ultralytics load failed: {e}")
                return False

        # Helper: try torchscript / torch.jit
        def try_torchscript():
            if not _HAS_TORCH:
                return False
            try:
                # torch.jit.load supports torchscript files; also torch.load on traced modules
                ts = torch.jit.load(mp, map_location='cpu')
                ts.eval()
                self.model = ts
                self.model_mode = 'torchscript'
                self.get_logger().info("Loaded model with torch.jit.load (torchscript).")
                return True
            except Exception as e:
                self.get_logger().warn(f"torchscript load failed: {e}")
                # try plain torch.load and accept state_dict only as raw fallback (not supported here)
                return False

        # selection logic
        if requested == 'ultralytics':
            if not try_ultralytics():
                self.get_logger().error("Requested ultralytics model but failed to load. Falling back to color_fallback.")
                self.model_mode = 'color_fallback'
        elif requested == 'torchscript':
            if not try_torchscript():
                self.get_logger().error("Requested torchscript model but failed to load. Falling back to color_fallback.")
                self.model_mode = 'color_fallback'
        else:  # auto
            if try_ultralytics():
                pass
            elif try_torchscript():
                pass
            else:
                self.get_logger().warn("Auto model load failed — using color-based fallback detector.")
                self.model_mode = 'color_fallback'

    # -------------------
    # Timer callback: capture, inference, publish
    # -------------------
    def _on_timer(self):
        now = self.get_clock().now().to_msg()

        # read frame
        frame = None
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn("Failed to read frame from camera this tick.")
                frame = None

        # If no frame, still publish synthetic centers (like your dummy) so system remains alive
        if frame is None:
            self.get_logger().debug("No frame available — publishing static centers and synthetic quality.")
            self._publish_static(now)
            self.k += 1
            return

        # run detection according to loaded model
        if self.model_mode == 'ultralytics':
            centers_mm, diam_mm, score = self._run_ultralytics_inference(frame)
        elif self.model_mode == 'torchscript':
            centers_mm, diam_mm, score = self._run_torchscript_inference(frame)
        else:
            # color based fallback that uses your detect_choux_from_frame logic (returns yolo-style bboxes)
            vis, good_cnts, yolo_labels = detect_choux_from_frame(frame)
            centers_mm, diam_mm, score = self._centers_from_yolo_labels(yolo_labels, frame.shape)

        # publish centers (convert to ROS msgs)
        msg_c = CupcakeCenters()
        msg_c.header.stamp = now
        msg_c.header.frame_id = self.frame_id
        msg_c.frame_id = self.frame_id
        # centers_mm expected as list of (x_m, y_m, z_m) or fallback: use stored self.centers
        if centers_mm and len(centers_mm) >= 3:
            # If model returned coordinates (x,y) in pixels or normalized, convert to physical positions
            # For now we simply publish the pre-configured centers if inference doesn't give real-world coords
            for (x, y, z) in self.centers:
                p = Point(); p.x = float(x); p.y = float(y); p.z = float(z)
                msg_c.centers.append(p)
        else:
            for (x, y, z) in self.centers:
                p = Point(); p.x = float(x); p.y = float(y); p.z = float(z)
                msg_c.centers.append(p)
        self.centers_pub.publish(msg_c)

        # publish quality
        msg_q = VisionQuality()
        msg_q.header.stamp = now
        # diam_mm expected as list length 3; fallback to synthetic wiggle
        if diam_mm and len(diam_mm) >= 3:
            msg_q.diameter_mm = [float(d) for d in diam_mm[:3]]
        else:
            d0 = self.diam_mean[0] + 1.0 * math.sin(self.k * 0.20)
            d1 = self.diam_mean[1] + 0.5 * math.sin(self.k * 0.15 + 1.0)
            d2 = self.diam_mean[2] + 0.7 * math.sin(self.k * 0.18 + 2.0)
            msg_q.diameter_mm = [float(d0), float(d1), float(d2)]
        # score fallback
        msg_q.score = [1.0, 1.0, 1.0] if score is None else score
        msg_q.needs_human = (max(msg_q.diameter_mm) - min(msg_q.diameter_mm)) > self.tol
        self.quality_pub.publish(msg_q)

        self.k += 1

    # -------------------
    # Ultralytics inference wrapper
    # -------------------
    def _run_ultralytics_inference(self, frame: np.ndarray) -> Tuple[List[Tuple[float,float,float]], List[float], Optional[List[float]]]:
        """
        Run ultralytics YOLO inference on the frame.
        Returns:
            centers_mm : placeholder list (empty) — you should replace with real mapping to world coords if available
            diam_mm    : list of diameters in mm (or synthetic)
            score      : optional list of confidences
        """
        try:
            # Ultralytics models accept numpy BGR frames directly
            results = self.model(frame, verbose=False)  # returns a Results object or list
            # results can be an iterator — pick first
            r = results[0] if isinstance(results, (list, tuple)) else results

            # bounding boxes: r.boxes or r.boxes.xyxy, r.boxes.conf, r.boxes.cls
            bboxes = []
            confidences = []
            if hasattr(r, 'boxes'):
                for box in r.boxes:
                    # box.xyxy gives tensor [x1,y1,x2,y2] in pixels
                    xyxy = box.xyxy.cpu().numpy().tolist()[0]
                    conf = float(box.conf.cpu().numpy()) if hasattr(box, 'conf') else 1.0
                    bboxes.append(xyxy)
                    confidences.append(conf)
            else:
                # ultralytics API changed: get .boxes.xyxy from r
                try:
                    xyxy_arr = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    for xyxy, conf in zip(xyxy_arr, confs):
                        bboxes.append(xyxy.tolist())
                        confidences.append(float(conf))
                except Exception:
                    pass

            # convert bboxes to simple outputs
            yolo_labels = []
            h, w = frame.shape[:2]
            for xyxy, conf in zip(bboxes, confidences):
                x1, y1, x2, y2 = xyxy
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                yolo_labels.append([0, cx / w, cy / h, bw, bh, conf])

            # derive diameters from bbox widths in pixels -> mm (placeholder mapping)
            diam_mm = []
            for lab in yolo_labels[:3]:
                _, cx_n, cy_n, bw, bh, conf = lab
                # naive diameter estimate: fraction of image width -> mm using a calibration_factor (tunable)
                calib_mm_per_frac = 400.0  # Placeholder: 400 mm for full image width — replace with calibration
                diam_mm.append(max(1.0, bw * calib_mm_per_frac))
            # pad to length 3
            while len(diam_mm) < 3:
                diam_mm.append(self.diam_mean[len(diam_mm)])
            return [], diam_mm, confidences[:3]
        except Exception as e:
            self.get_logger().error(f"Ultralytics inference error: {e}")
            return [], [], None

    # -------------------
    # Torchscript inference wrapper (very generic)
    # -------------------
    def _run_torchscript_inference(self, frame: np.ndarray) -> Tuple[List[Tuple[float,float,float]], List[float], Optional[List[float]]]:
        if not _HAS_TORCH or self.model is None:
            return [], [], None
        try:
            # convert image to CHW float tensor normalized 0..1
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            h, w = img.shape[:2]
            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # 1xCxHxW
            with torch.no_grad():
                out = self.model(tensor)  # expected to be a tensor or (boxes, scores, classes)
            # Best effort parsing: if out is (N,6) with xyxy and conf in last column
            boxes = []
            confs = []
            if isinstance(out, torch.Tensor):
                arr = out.cpu().numpy()
                # try to interpret rows: [x1,y1,x2,y2,conf,class] or similar
                if arr.ndim == 2 and arr.shape[1] >= 5:
                    for row in arr:
                        if row[4] < 0.01:
                            continue
                        x1, y1, x2, y2 = row[:4]
                        conf = float(row[4])
                        boxes.append([x1, y1, x2, y2])
                        confs.append(conf)
            # convert to yolo labels normalized
            yolo_labels = []
            for xyxy, conf in zip(boxes, confs):
                x1, y1, x2, y2 = xyxy
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                bw = (x2 - x1) / float(w)
                bh = (y2 - y1) / float(h)
                yolo_labels.append([0, cx / w, cy / h, bw, bh, conf])
            # map to diam_mm as in ultralytics wrapper
            diam_mm = []
            calib_mm_per_frac = 400.0
            for lab in yolo_labels[:3]:
                _, cx_n, cy_n, bw, bh, conf = lab
                diam_mm.append(max(1.0, bw * calib_mm_per_frac))
            while len(diam_mm) < 3:
                diam_mm.append(self.diam_mean[len(diam_mm)])
            return [], diam_mm, confs[:3]
        except Exception as e:
            self.get_logger().error(f"Torchscript inference error: {e}")
            return [], [], None

    # -------------------
    # helper: map yolo labels to center/diam (placeholder)
    # -------------------
    def _centers_from_yolo_labels(self, yolo_labels, img_shape):
        """
        Convert normalized YOLO labels to dummy centers/diameters.
        This function is intentionally simple — replace with your calibration (pixel->world) mapping.
        """
        h, w = img_shape[:2]
        diam_mm = []
        for lab in yolo_labels[:3]:
            _, cx, cy, bw, bh = lab[:5]
            # naive diameter estimate
            calib_mm_per_frac = 400.0
            diam_mm.append(max(1.0, bw * calib_mm_per_frac))
        while len(diam_mm) < 3:
            diam_mm.append(self.diam_mean[len(diam_mm)])
        return [], diam_mm, None

    # -------------------
    # If no frame: publish original static centers & synthetic quality
    # -------------------
    def _publish_static(self, now):
        msg_c = CupcakeCenters()
        msg_c.header.stamp = now
        msg_c.header.frame_id = self.frame_id
        msg_c.frame_id = self.frame_id
        for (x, y, z) in self.centers:
            p = Point(); p.x = float(x); p.y = float(y); p.z = float(z)
            msg_c.centers.append(p)
        self.centers_pub.publish(msg_c)

        msg_q = VisionQuality()
        msg_q.header.stamp = now
        d0 = self.diam_mean[0] + 1.0 * math.sin(self.k * 0.20)
        d1 = self.diam_mean[1] + 0.5 * math.sin(self.k * 0.15 + 1.0)
        d2 = self.diam_mean[2] + 0.7 * math.sin(self.k * 0.18 + 2.0)
        msg_q.diameter_mm = [float(d0), float(d1), float(d2)]
        msg_q.score = [1.0, 1.0, 1.0]
        msg_q.needs_human = (max(msg_q.diameter_mm) - min(msg_q.diameter_mm)) > self.tol
        self.quality_pub.publish(msg_q)

    def destroy_node(self):
        # release camera
        if self.cap and self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

def main():
    rclpy.init()
    node = VisionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


# -------------------
# Keep your existing helper functions for color-based detection
# -------------------
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
    h, w = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    S_min = 50
    V_min = 80
    mask_sv = (S > S_min) & (V > V_min)
    mask = mask_sv.astype(np.uint8) * 255

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

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
    # optional display removed in node context

    yolo_labels = []
    for c in good:
        x, y, ww, hh = cv2.boundingRect(c)
        cx, cy = x + ww / 2, y + hh / 2
        cls = 0
        yolo_labels.append([cls, cx / w, cy / h, ww / w, hh / h])
    return vis, good, yolo_labels
