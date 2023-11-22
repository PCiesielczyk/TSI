import logging
import os

import matplotlib.pyplot as plt
import neptune
import numpy as np
import pandas as pd
from PIL import Image
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from neptune.integrations.tensorflow_keras import NeptuneCallback
from neptune.types import File
from sklearn.metrics import precision_score, recall_score, confusion_matrix

from TSI.data_storage.file_loader import X_train_aug, y_train_aug, X_val, y_val

logging.basicConfig(level=logging.INFO)

module_dir = os.path.dirname(os.getcwd())
project_dir = os.path.dirname(module_dir)
dataset_dir = 'archive'
train_dir = os.path.join(project_dir, dataset_dir, 'Train')
meta_dir = os.path.join(project_dir, dataset_dir, 'Meta')
class_count = len(np.unique(y_train_aug))

logging.info(f'Training set X shape: {X_train_aug.shape}, y shape: {y_train_aug.shape}')
logging.info(f'Validation set X shape: {X_val.shape}, y shape: {y_val.shape}')

run = neptune.init_run()
neptune_callback = NeptuneCallback(run=run, log_model_diagram=True)
run["train_dataset/images"].track_files(f"file://{train_dir}")

for filename in os.listdir(meta_dir):
    if filename[0].isdigit():
        run["meta/images"].append(File.as_image(Image.open(os.path.join(meta_dir, filename))))

# Converting the labels into one hot encoding
y_train = to_categorical(y_train_aug, class_count)
y_val = to_categorical(y_val, class_count)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train_aug.shape[1:]))
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
model.add(Dense(class_count, activation='softmax'))

print(model.summary())

# Compilation of the models
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

eps = 40
history = model.fit(X_train_aug, y_train, batch_size=256, epochs=eps, validation_data=(X_val, y_val),
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
    image = image.resize((32, 32))
    data.append(np.array(image))
X_test = np.array(data)
y_test = to_categorical(y_test, class_count)

logging.info(f'Test set X shape: {X_test.shape}, y shape: {y_test.shape}')
test_loss, test_accuracy = model.evaluate(X_test, y_test)

logging.info(f'Test Loss: {test_loss}')
logging.info(f'Test Accuracy: {test_accuracy}')

run["test_preds/test_loss"] = str(test_loss)
run["test_preds/test_accuracy"] = str(test_accuracy)

y_pred = model.predict(X_test)
y_true = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

precision = precision_score(y_true, y_pred_labels, average='macro')
recall = recall_score(y_true, y_pred_labels, average='macro')

logging.info(f'Precision: {precision}')
logging.info(f'Recall: {recall}')

run["test_preds/precision"] = str(precision)
run["test_preds/recall"] = str(recall)

confusion = confusion_matrix(np.argmax(y_test, axis=1), y_pred_labels)

fig, ax = plt.subplots(figsize=(8, 8))

# Create the heatmap
heatmap = ax.imshow(confusion, cmap='Blues')
ax.set_xlabel('Estimated')
ax.set_ylabel('Ground truth')
cbar = ax.figure.colorbar(heatmap, ax=ax)

plt.show()

run.stop()
