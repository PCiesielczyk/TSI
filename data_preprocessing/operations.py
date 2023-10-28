import numpy as np
import cv2
import logging
from skimage.transform import rotate
from skimage.transform import warp
from skimage.transform import ProjectiveTransform

logging.basicConfig(level=logging.INFO)


def rotate_image(image, max_angle=15):
    rotate_out = rotate(image, np.random.uniform(-max_angle, max_angle), mode='edge')
    return rotate_out


def translate_image(image, max_trans=5, height=32, width=32):
    translate_x = max_trans * np.random.uniform() - max_trans / 2
    translate_y = max_trans * np.random.uniform() - max_trans / 2
    translation_mat = np.float32([[1, 0, translate_x], [0, 1, translate_y]])
    trans = cv2.warpAffine(image, translation_mat, (height, width))
    return trans


def projection_transform(image, max_warp=0.8, height=32, width=32):
    # Warp Location
    d = height * 0.3 * np.random.uniform(0, max_warp)

    # Warp co-ordinates
    tl_top = np.random.uniform(-d, d)  # Top left corner, top margin
    tl_left = np.random.uniform(-d, d)  # Top left corner, left margin
    bl_bottom = np.random.uniform(-d, d)  # Bottom left corner, bottom margin
    bl_left = np.random.uniform(-d, d)  # Bottom left corner, left margin
    tr_top = np.random.uniform(-d, d)  # Top right corner, top margin
    tr_right = np.random.uniform(-d, d)  # Top right corner, right margin
    br_bottom = np.random.uniform(-d, d)  # Bottom right corner, bottom margin
    br_right = np.random.uniform(-d, d)  # Bottom right corner, right margin

    # Apply Projection
    transform = ProjectiveTransform()
    transform.estimate(np.array((
        (tl_left, tl_top),
        (bl_left, height - bl_bottom),
        (height - br_right, height - br_bottom),
        (height - tr_right, tr_top)
    )), np.array((
        (0, 0),
        (0, height),
        (height, height),
        (height, 0)
    )))
    output_image = warp(image, transform, output_shape=(height, width), order=1, mode='edge')
    return output_image


def transform_image(image, max_angle=15, max_trans=5, max_warp=0.8):

    height, width, channels = image.shape
    # Rotate Image
    rotated_image = rotate_image(image, max_angle)
    # Translate Image
    translated_image = translate_image(rotated_image, max_trans, height, width)
    # Project Image
    output_image = projection_transform(translated_image, max_warp, height, width)
    return (output_image * 255.0).astype(np.uint8)


def augment_and_balance_data(X_train_data, y_train_data, no_examples_per_class):
    n_examples = no_examples_per_class
    # Get parameters of data
    classes, class_indices, class_counts = np.unique(y_train_data, return_index=True, return_counts=True)
    height, width, channels = X_train_data[0].shape

    # Create new data and labels for the balanced augmented data
    X_balance = np.empty([0, X_train_data.shape[1], X_train_data.shape[2], X_train_data.shape[3]], dtype=np.float32)
    y_balance = np.empty([0], dtype=y_train_data.dtype)

    for c, count in zip(range(len(classes)), class_counts):
        # Copy over the current data for the given class
        X_orig = X_train_data[y_train_data == c]
        y_orig = y_train_data[y_train_data == c]
        # Add original data to the new dataset
        X_balance = np.append(X_balance, X_orig, axis=0)
        logging.info(f'Augmenting {c} class with {n_examples - count} samples')
        temp_X = np.empty([n_examples - count, X_train_data.shape[1], X_train_data.shape[2], X_train_data.shape[3]], dtype=np.float32)
        for i in range(n_examples - count):
            temp_X[i, :, :, :] = transform_image(X_orig[i % count]).reshape((1, height, width, channels))

        X_balance = np.append(X_balance, temp_X, axis=0)
        n_added_ex = X_balance.shape[0] - y_balance.shape[0]
        y_balance = np.append(y_balance, np.full(n_added_ex, c, dtype=int))

    logging.info(f'Augmented training set X shape: {X_balance.shape}, y shape: {y_balance.shape}')
    return X_balance.astype(np.uint8), y_balance
