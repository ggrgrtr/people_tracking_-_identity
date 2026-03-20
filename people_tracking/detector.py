import cv2
from ultralytics import YOLO

from .utils import (
    bbox_area,
    box_valid,
    clip_bbox,
    compute_iou,
    get_center,
    intersection_over_smaller,
    point_in_bbox,
    resolve_yolo_weights,
)


class PersonDetector:
    def __init__(self, config, base_dir, device):
        self.config = config
        self.model = YOLO(resolve_yolo_weights(base_dir))

        try:
            self.model.to(device)
        except Exception:
            pass

    def _suppress_duplicate_detections(self, detections):
        if len(detections) <= 1:
            return [bbox for bbox, _ in detections]

        ordered = sorted(
            detections,
            key=lambda item: (bbox_area(item[0]), item[1]),
            reverse=True,
        )
        filtered = []

        for bbox, confidence in ordered:
            center = get_center(bbox)
            duplicate = False

            for kept_bbox, _ in filtered:
                iou = compute_iou(kept_bbox, bbox)
                nested = intersection_over_smaller(kept_bbox, bbox)

                if iou >= self.config.detector_duplicate_iou:
                    duplicate = True
                    break

                if (
                    nested >= self.config.detector_nested_overlap
                    and point_in_bbox(center, kept_bbox)
                ):
                    duplicate = True
                    break

            if not duplicate:
                filtered.append((bbox, confidence))

        return [bbox for bbox, _ in filtered]

    def detect(self, frame):
        frame_h, frame_w = frame.shape[:2]
        scaled_w = max(64, int(frame_w * self.config.yolo_scale))
        scaled_h = max(64, int(frame_h * self.config.yolo_scale))
        scaled = cv2.resize(frame, (scaled_w, scaled_h))

        results = self.model(
            scaled,
            imgsz=self.config.yolo_imgsz,
            conf=self.config.yolo_conf,
            classes=[0],
            verbose=False,
        )

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1 = int(x1 / self.config.yolo_scale)
                y1 = int(y1 / self.config.yolo_scale)
                x2 = int(x2 / self.config.yolo_scale)
                y2 = int(y2 / self.config.yolo_scale)

                bbox = clip_bbox((x1, y1, x2 - x1, y2 - y1), frame.shape)
                if box_valid(
                    bbox,
                    frame.shape,
                    self.config.min_box_w,
                    self.config.min_box_h,
                ):
                    confidence = float(box.conf[0]) if box.conf is not None else 0.0
                    detections.append((bbox, confidence))

        return self._suppress_duplicate_detections(detections)
