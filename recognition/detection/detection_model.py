import torch
from recognition.detection.detected_traffic_sign import DetectedTrafficSign


class DetectionModel:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='recognition/detection/yolov5_weights.pt')

    def detect_traffic_signs(self, image_path) -> list[DetectedTrafficSign]:
        traffic_signs = []
        results = self.model(image_path)

        for index, row in results.pandas().xyxy[0].iterrows():
            traffic_signs.append(DetectedTrafficSign(row['xmin'], row['ymin'], row['xmax'],
                                                     row['ymax'], row['confidence']))

        return traffic_signs
