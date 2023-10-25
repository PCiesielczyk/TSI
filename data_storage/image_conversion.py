import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split

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
            image = image.resize((32, 32))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")

data = np.array(data)
labels = np.array(labels)

X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)
np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)
