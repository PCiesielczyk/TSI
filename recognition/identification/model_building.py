import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import neptune
from PIL import Image
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from neptune.integrations.tensorflow_keras import NeptuneCallback
from neptune.types import File
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

module_dir = os.path.dirname(os.getcwd())
project_dir = os.path.dirname(module_dir)
dataset_dir = 'archive'
data_storage_dir = 'data_storage'
train_dir = os.path.join(project_dir, dataset_dir, 'Train')
meta_dir = os.path.join(project_dir, dataset_dir, 'Meta')

data_file_path = os.path.join(project_dir, data_storage_dir, 'data.npy')
labels_file_path = os.path.join(project_dir, data_storage_dir, 'labels.npy')

data = np.load(data_file_path)
labels = np.load(labels_file_path)

print(data.shape, labels.shape)

# Splitting training and testing dataset
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

run = neptune.init_run()
neptune_callback = NeptuneCallback(run=run, log_model_diagram=True)
run["train_dataset/images"].track_files(f"file://{train_dir}")

for filename in os.listdir(meta_dir):
    if filename[0].isdigit():
        run["meta/images"].append(File.as_image(Image.open(os.path.join(meta_dir, filename))))

# Converting the labels into one hot encoding
y_train = to_categorical(y_train, 43)
y_val = to_categorical(y_val, 43)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

print(model.summary())

# Compilation of the models
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

eps = 15
history = model.fit(X_train, y_train, batch_size=32, epochs=eps, validation_data=(X_val, y_val),
                    callbacks=[neptune_callback])

# plotting graphs for accuracy
plt.figure(figsize=(16, 5))
plt.subplot(121)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('models accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('models loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')

plt.show()

model.save('TSI_model.h5')
run["saved_model"].upload("TSI_model.h5")

test_file = 'Test.csv'
test_csv_data = pd.read_csv(os.path.join(project_dir, dataset_dir, test_file))
y_test = np.array(test_csv_data["ClassId"].values)
images = test_csv_data["Path"].values
data = []

for img in images:
    image = Image.open(os.path.join(project_dir, dataset_dir, img).replace('/', '\\'))
    image = image.resize((30, 30))
    data.append(np.array(image))
X_test = np.array(data)
y_test = to_categorical(y_test, 43)

print(X_test.shape)
print(y_test.shape)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

y_pred = model.predict(X_test)
y_true = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

precision = precision_score(y_true, y_pred_labels, average='macro')
recall = recall_score(y_true, y_pred_labels, average='macro')

print("Precision:", precision)
print("Recall:", recall)

run.stop()
