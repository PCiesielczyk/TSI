class DetectedTrafficSign:
    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float, confidence):
        self.x_min = round(x_min)
        self.y_min = round(y_min)
        self.x_max = round(x_max)
        self.y_max = round(y_max)
        self.confidence = confidence
