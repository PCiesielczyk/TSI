from TSI.recognition.identification.class_mapper import ClassMapper


class Prediction:
    def __init__(self, class_id, probability):
        self.class_id = class_id
        self.probability = probability
        self.sign_name = ClassMapper().map_to_text(class_id)
