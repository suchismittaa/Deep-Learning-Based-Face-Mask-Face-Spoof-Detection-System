import cv2
import numpy as np
from mtcnn import MTCNN


class FaceDetector:
    """Detect the largest face and crop it with a small contextual margin."""

    def __init__(self, margin: float = 0.20):
        self.detector = MTCNN()
        self.margin = margin

    def detect_and_crop(self, image_rgb: np.ndarray):
        if image_rgb is None or image_rgb.size == 0:
            return None

        results = self.detector.detect_faces(image_rgb)
        if not results:
            return None

        largest_face = max(results, key=lambda r: max(0, r["box"][2]) * max(0, r["box"][3]))
        x, y, w, h = largest_face["box"]
        x, y = max(0, x), max(0, y)
        w, h = max(1, w), max(1, h)

        x1 = max(0, int(x - self.margin * w))
        y1 = max(0, int(y - self.margin * h))
        x2 = min(image_rgb.shape[1], int(x + w + self.margin * w))
        y2 = min(image_rgb.shape[0], int(y + h + self.margin * h))

        if x2 <= x1 or y2 <= y1:
            return None
        return image_rgb[y1:y2, x1:x2]
