# Traffic Signs Identifier
### A tool for detecting and classifying traffic signs from image and web camera. It also includes utensils for preparation and preprocessing of dataset and model training.

## Data storage
In order to process the data correctly, first the dataset needs to be converted from raw images to `numpy.ndarray` format. This is performed by `data_storage/image_conversion.py` script. The dataset structure is as follows:
```
.
└── TSI/
    └── archive/
        ├── classes_map.csv
        ├── Test.csv
        ├── Meta/
        │   ├── 0.jpg
        │   ├── 1.jpg
        │   └── ...
        ├── Train/
        │   ├── 0/
        │   │   ├── 00000_00000_00000.png
        │   │   ├── 00000_00000_00001.png
        │   │   └── ...
        │   ├── 1/
        │   │   ├── 00001_00000_00000.png
        │   │   ├── 00001_00000_00001.png
        │   │   └── ...
        │   └── ...
        └── Test/
            ├── 00000.png
            ├── 00001.png
            └── ...
```
This structure and test samples' annotations in `Test.csv` file are based on [GTSRB Dataset](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign). The only difference is `classes_map.csv` which contains information about classes ids and names. For example:
```
0,Speed limit (40km/h)
1,No left turn
2,Priority road
...
```
`image_conversion.py` converts all train images to `numpy.ndarray` format, splits data to train and validation subsets and saves them to `X_train.npy`, `X_val.npy`, `y_train.npy` and `y_val.npy` files.

### Utensils
To edit the dataset more conveniently, two scripts can be used:
- `data_storage/class_remover.py` takes class ids as an argument (--class_no) and remove all related data (train/test samples, meta, test entries) and decrements the class id that follows it.
- `data_storage/class_shifter.py` takes input class id (--in_class) and id to rename it (--out_class). All other class ids shift accordingly.

## Data preprocessing
Preprocessing focuses on two operations - undersampling and augmentation that minimize the effects of unbalanced class distribution.
- undersampling removes random data from majority class using RandomUnderSampler module from imblearn. To perform undersampling run `data_preprocessing/undersampling.py` and to set maximum number of samples for each class change the `samples_count_threshold` variable. 
By default it is set to 500. After undersampling the data is saved in `X_train_undersampled.npy` and `y_train_undersampled.npy` files.
- augmentation aplies various transformations on minority classes to increase its size. Details about them can be found on [Deep-traffic-sign-classification](https://github.com/joshwadd/Deep-traffic-sign-classification) repo in Data Augmentation chapter.
To generate augmented data run `data_preprocessing/augmentation.py` and to set the minimum number of samples for each class change the `samples_after_aug` variable. By default it is set to 500. After augmentation the data is saved in `X_train_aug.npy` and `y_train_aug.npy` files.
Make sure to run undersampling first as augmentation is performed on unsersampled data files.

## Identification model training
Traffic signs classifying model is based on convolution layers and trained using keras api. To start training execute `recognition/identification/model_building.py`. This file contains model layers' structure and also performs tests after model is built. 
Trained model weights are saved in `recognition/identification/TSI_model.h5` file.

Note that training is integrated with the [Neptune](https://neptune.ai/) platform. It is a tool that enables tracking the results of models and dataset structure. 
To properly log model data, make sure that the NEPTUNE_API_TOKEN and NEPTUNE_PROJECT environment variables are set. This can be obtained by setting up a project on Neptune platform. To disable integration remove Neptune callback from `model_building.py`.

## Identification from image
To perform identification the traffic signs detection model weights must be also included. This can be done by [training YOLOv5 model on custom data](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/). 
To prepare dataset use [DFG to YOLOv5 converter](https://github.com/PCiesielczyk/DFG_to_yolov5_converter). Save the resulting weights under the `recognition/detection/yolov5_weights.pt` path.
Identifying traffic signs from an image is performed with simple GUI that is initialized with `TSI_from_image.py`. Pass the image with `Upload an image` button and set probability threshold (0 - 100%) in right upper corner - this can filter out uncertain predictions. 
Then press `Classify Image` to identify traffic signs.
> The detection model's confidence can also be adjust using `confidence_threshold` in executed script. By default it is set to 0.6
<img src="/images/TSI_image.png" alt="TSI_image">

## Identification from web camera
Identifying traffic signs from a web camera is performed with `TSI_from_webcam.py`. It will open a window capturing the image from the camera and perform identification on each frame. If many camera output are available they can be chose by `cam_port` variable.
> The identification probablity threshold can be adjusted with `prob_threshold` variable and detection model's confidence with `confidence_threshold`.
<img src="/images/TSI_cam.png" alt="TSI_cam">
