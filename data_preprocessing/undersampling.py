import os
import logging
import numpy as np
from TSI.data_storage.file_loader import X_train, y_train
from imblearn.under_sampling import RandomUnderSampler
import csv

logging.basicConfig(level=logging.INFO)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.getcwd())

samples_count_threshold = 500

unique_train, counts_train = np.unique(y_train, return_counts=True)
sampling_strategy = {}
new_col_values = []
for class_id in unique_train:
    class_samples_count = counts_train[class_id]

    new_col_values.append(class_samples_count)

    if class_samples_count > samples_count_threshold:
        sampling_strategy[class_id] = samples_count_threshold
    else:
        sampling_strategy[class_id] = class_samples_count
    logging.info(f'Setting class {class_id} to {sampling_strategy[class_id]} samples')

rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
print(X_train.shape, y_train.shape)

X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
y_train_reshaped = y_train.reshape(-1, 1)

X_resampled, y_resampled = rus.fit_resample(X_train_reshaped, y_train_reshaped)
X_resampled_original_shape = X_resampled.reshape(X_resampled.shape[0], 32, 32, 3)

np.save('X_train_undersampled.npy', X_resampled_original_shape)
np.save('y_train_undersampled.npy', y_resampled)

os.rename(os.path.join(current_dir, 'X_train_undersampled.npy'), os.path.join(project_dir, 'data_storage',
                                                                              'X_train_undersampled.npy'))
os.rename(os.path.join(current_dir, 'y_train_undersampled.npy'), os.path.join(project_dir, 'data_storage',
                                                                              'y_train_undersampled.npy'))
