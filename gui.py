import logging
import os
import tkinter as tk
from tkinter import *
from tkinter import filedialog

import numpy as np
from PIL import ImageTk, Image, ImageDraw, ImageFont
from keras.models import load_model

from recognition.detection.detection_model import DetectionModel
from recognition.identification.class_mapper import ClassMapper

logging.basicConfig(level=logging.INFO)

# identification model
project_dir = os.getcwd()
identification_model_weights_path = os.path.join(project_dir, 'recognition/identification/TSI_model.h5')
identification_model = load_model(identification_model_weights_path)

# detection model
detection_model = DetectionModel()

# initialise GUI
top = tk.Tk()
top.geometry('1300x900')
top.title('Traffic sign classification')
top.configure(background='#CDCDCD')
label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)
class_mapper = ClassMapper()


def classify(file_path):
    image = Image.open(file_path)
    preview_image = Image.open(file_path)
    image_draw = ImageDraw.Draw(preview_image)
    font = ImageFont.truetype("arial.ttf", 34)
    traffic_signs = detection_model.detect_traffic_signs(file_path)

    for traffic_sign in traffic_signs:
        x1, y1 = traffic_sign.x_min, traffic_sign.y_min
        x2, y2 = traffic_sign.x_max, traffic_sign.y_max

        cropped_image = image.crop((x1, y1, x2, y2))
        cropped_image = cropped_image.resize((30, 30))
        cropped_image = np.expand_dims(cropped_image, axis=0)
        cropped_image = np.array(cropped_image)
        pred = identification_model.predict([cropped_image])
        pred_class = np.argmax(pred, axis=1)[0]
        probability = np.max(pred, axis=1)[0] * 100
        sign = class_mapper.map_to_text(pred_class + 1)

        output_text = f"{sign} with {probability:.4f}% probability\n"
        print(output_text)

        image_draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=2)
        image_draw.text((x1, y2 + 15), sign, fill="red", font=font)

    update_preview_image(preview_image)


def update_preview_image(image):
    image.thumbnail((round((top.winfo_width() / 1.3)), round((top.winfo_height() / 1.3))))
    marked_image = ImageTk.PhotoImage(image)
    sign_image.configure(image=marked_image)
    sign_image.image = marked_image


def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        update_preview_image(uploaded)
        label.configure(text='')
        show_classify_button(file_path)
    except:
        logging.error('Uploading image failed')


upload = Button(top, text="Upload an image", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
heading = Label(top, text="check traffic sign", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
top.mainloop()
