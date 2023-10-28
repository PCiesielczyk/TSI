import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

X_train = np.load(os.path.join(current_dir, 'X_train.npy'))
X_val = np.load(os.path.join(current_dir, 'X_val.npy'))
y_train = np.load(os.path.join(current_dir, 'y_train.npy'))
y_val = np.load(os.path.join(current_dir, 'y_val.npy'))

X_train_aug = np.load(os.path.join(current_dir, 'X_train_aug.npy'))
y_train_aug = np.load(os.path.join(current_dir, 'y_train_aug.npy'))
