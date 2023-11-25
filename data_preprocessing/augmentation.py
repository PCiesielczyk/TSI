import os

import matplotlib.pyplot as plt
import numpy as np

from TSI.data_preprocessing.augment_operations import augment_and_balance_data
from TSI.data_storage.file_loader import y_train, X_train_undersampled, y_train_undersampled

samples_after_aug = 500

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.getcwd())

plt.figure(0, figsize=(10, 5))
unique_train, counts_train = np.unique(y_train, return_counts=True)
plt.bar(unique_train, counts_train)

plt.title('Training Set Class Distribution before under-sampling', fontsize=22)
plt.xlabel('Class Number', fontsize=18)
plt.ylabel('Number of Occurrences', fontsize=20)
plt.tick_params(labelsize=16)
plt.grid(linestyle=':')
plt.show()

plt.figure(1, figsize=(10, 5))
unique_train_us, counts_train_us = np.unique(y_train_undersampled, return_counts=True)
plt.bar(unique_train_us, counts_train_us)

plt.title('Training Set Class Distribution after under-sampling', fontsize=22)
plt.xlabel('Class Number', fontsize=18)
plt.ylabel('Number of Occurrences', fontsize=20)
plt.tick_params(labelsize=16)
plt.grid(linestyle=':')
plt.show()


X_train_aug, y_train_aug = augment_and_balance_data(X_train_undersampled, y_train_undersampled, samples_after_aug)
np.save('X_train_aug.npy', X_train_aug)
np.save('y_train_aug.npy', y_train_aug)

os.rename(os.path.join(current_dir, 'X_train_aug.npy'), os.path.join(project_dir, 'data_storage', 'X_train_aug.npy'))
os.rename(os.path.join(current_dir, 'y_train_aug.npy'), os.path.join(project_dir, 'data_storage', 'y_train_aug.npy'))

plt.figure(1, figsize=(10, 5))
unique_train_aug, counts_train_aug = np.unique(y_train_aug, return_counts=True)
plt.bar(unique_train_aug, counts_train_aug)

plt.title('Training Set Class Distribution after augmentation', fontsize=22)
plt.xlabel('Class Number', fontsize=18)
plt.ylabel('Number of Occurrences', fontsize=20)
plt.tick_params(labelsize=16)
plt.grid(linestyle=':')
plt.show()
