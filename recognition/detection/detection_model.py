import os
import torch
from TSI.recognition.detection.detected_traffic_sign import DetectedTrafficSign


class DetectionModel:
    def __init__(self, confidence_threshold):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(current_dir, 'yolov5_weights.pt')
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
        self.model.conf = confidence_threshold

    def detect_traffic_signs(self, image) -> list[DetectedTrafficSign]:
        traffic_signs = []
        results = self.model(image)

        for index, row in results.pandas().xyxy[0].iterrows():
            traffic_signs.append(DetectedTrafficSign(row['xmin'], row['ymin'], row['xmax'],
                                                     row['ymax'], row['confidence']))

        return traffic_signs
