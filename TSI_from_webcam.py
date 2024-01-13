import cv2
import logging
import numpy as np
from TSI.recognition.detection.detection_model import DetectionModel
from TSI.recognition.identification.identification_model import IdentificationModel

logging.basicConfig(level=logging.INFO)

detection_model = DetectionModel(confidence_threshold=0.6)
identification_model = IdentificationModel()

cam_port = 0
cam = cv2.VideoCapture(cam_port)
cam.set(3, 1400)
cam.set(4, 1400)

prob_threshold = 40

rectangle_thickness = 1
red_RGB = (255, 0, 0)
font_scale = 0.7
font_thickness = 1

while True:
    result, image = cam.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    traffic_signs = detection_model.detect_traffic_signs(image)

    for traffic_sign in traffic_signs:
        x1, y1 = traffic_sign.x_min, traffic_sign.y_min
        x2, y2 = traffic_sign.x_max, traffic_sign.y_max

        cropped_image = image[y1:y2, x1:x2]
        cropped_image = cv2.resize(cropped_image, (32, 32))
        cropped_image = np.expand_dims(cropped_image, axis=0)
        cropped_image = np.array(cropped_image)
        prediction = identification_model.predict([cropped_image])

        output_text = f"{prediction.sign_name} with {prediction.probability:.4f}% probability\n"
        print(output_text)

        if prediction.probability >= prob_threshold:
            image = cv2.rectangle(image, (x1, y1), (x2, y2), red_RGB, 1)
            image = cv2.putText(image, prediction.sign_name, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                red_RGB, font_thickness, cv2.LINE_AA)

    output_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Webcam output", output_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
