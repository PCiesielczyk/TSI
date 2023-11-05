from class_mapper import ClassMapper


class Prediction:
    def __init__(self, class_no, probability):
        self.class_no = class_no
        self.probability = probability
        self.sign_name = ClassMapper().map_to_text(class_no + 1)
