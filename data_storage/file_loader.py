import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = os.path.dirname(current_dir)
archive_dir = os.path.join(module_dir, 'archive')

classes_csv_file_path = os.path.join(archive_dir, 'classes_map.csv')
train_dir_path = os.path.join(archive_dir, 'Train')

X_train_aug_path = os.path.join(current_dir, 'X_train_aug.npy')
y_train_aug_path = os.path.join(current_dir, 'y_train_aug.npy')

X_train_undersampled_path = os.path.join(current_dir, 'X_train_undersampled.npy')
y_train_undersampled_path = os.path.join(current_dir, 'y_train_undersampled.npy')

X_train = np.load(os.path.join(current_dir, 'X_train.npy'))
X_val = np.load(os.path.join(current_dir, 'X_val.npy'))
y_train = np.load(os.path.join(current_dir, 'y_train.npy'))
y_val = np.load(os.path.join(current_dir, 'y_val.npy'))

if os.path.exists(X_train_aug_path):
    X_train_aug = np.load(X_train_aug_path)

if os.path.exists(y_train_aug_path):
    y_train_aug = np.load(y_train_aug_path)

if os.path.exists(X_train_undersampled_path):
    X_train_undersampled = np.load(X_train_undersampled_path)

if os.path.exists(y_train_undersampled_path):
    y_train_undersampled = np.load(y_train_undersampled_path)
