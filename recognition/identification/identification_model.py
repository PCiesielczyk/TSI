import os
import numpy as np
from keras.models import load_model
from TSI.recognition.identification.prediction import Prediction


class IdentificationModel:
    def __init__(self):
        self._current_dir = os.path.dirname(os.path.abspath(__file__))
        self._identification_model_weights_path = os.path.join(self._current_dir, 'TSI_model.h5')
        self._model = load_model(self._identification_model_weights_path)

    def predict(self, image_array: np.ndarray) -> Prediction:
        model_prediction = self._model.predict([image_array])
        pred_class = np.argmax(model_prediction, axis=1)[0]
        probability = np.max(model_prediction, axis=1)[0] * 100
        return Prediction(pred_class, probability)

