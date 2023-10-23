import numpy as np
from PIL import Image
import os

data = []
labels = []
project_dir = os.path.dirname(os.getcwd())
train_dir = os.path.join(project_dir, 'archive', 'Train')
classes = len(os.listdir(train_dir))

for i in range(classes):
    path = os.path.join(train_dir, str(i))
    images = os.listdir(path)

    print(f"Processing {i} class")

    for filename in images:
        image_path = os.path.join(path, filename)

        try:
            image = Image.open(image_path)
            print(image_path)
            image = image.resize((32, 32))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")

data = np.array(data)
labels = np.array(labels)

np.save('data.npy', data)
np.save('labels.npy', labels)
