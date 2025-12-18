import cv2
import numpy as np
from mtcnn import MTCNN

class FaceDetector:
    def __init__(self):
        self.detector = MTCNN()
    
    def detect_and_crop(self, image_rgb):
        results = self.detector.detect_faces(image_rgb)
        if not results:
            return None
            
        largest_face = max(results, key=lambda r: r['box'][2] * r['box'][3])
        x, y, w, h = largest_face['box']
        x, y = abs(x), abs(y)
        
        margin = 0.2
        x_m = int(x - margin * w)
        y_m = int(y - margin * h)
        w_m = int(w * (1 + 2 * margin))
        h_m = int(h * (1 + 2 * margin))
        
        img_h, img_w, _ = image_rgb.shape
        x1, y1 = max(0, x_m), max(0, y_m)
        x2, y2 = min(img_w, x_m + w_m), min(img_h, y_m + h_m)
        
        cropped_face = image_rgb[y1:y2, x1:x2]
        return cropped_face