"""
Udacity dataset general functions.

These are adopted in training, testing the models and also driving the
Udacity simulator autonomous car.
"""
import os
import random
import numpy as np
import pandas as pd
import csv
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from constants import IMAGE_HEIGHT, IMAGE_CHANNELS, IMAGE_WIDTH, TRAIN_DATA_DIR, B_SIZE, NB_EPOCHS, LR


def load_data(labels_fl, test_size=None):
    """
    Load and split the data into training and testing sets.


    :param labels_fl: directory of the labels (measurement logs) CSV file
    :param test_size: size of the testing set
    :return: training and testing input and output sets if a test_size parameter is provided
    otherwise return without splitting
    """

    labels = pd.read_csv(labels_fl)
    x = labels[['center', 'left', 'right']].values
    y = labels['steering'].values

    # R
    if test_size:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
        return x_train, x_test, y_train, y_test
    return x, y


def load_image(dt_dir, image_file):
    """
    Load an RGB image.

    :param dt_dir: Images directory
    :param image_file: Image file name
    :return: RGB image
    """

    return mpimg.imread(os.path.join(dt_dir, image_file.strip()))


def preprocess(img):
    """
    Preprocess the input image by cropping, resizing and converting to YUV colour space.

    :param img: input image
    :return: preprocessed image
    """

    # Crop the image
    img = img[60:-25, :, :]

    # Resize the image
    img = cv2.resize(img, (200, 66), cv2.INTER_AREA)

    # Convert the image to YUV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    return img


# Data augmentation functions

def random_adjust(data_dir, center, left, right, steering_angle):
    """
    Adjust the steering angle of a random image.

    :param data_dir: images directory
    :param center: center view image
    :param left: left view image
    :param right: right view image
    :param steering_angle: the steering angle related to the input frame
    :return: random image and its corresponding steering angle after adjustment
    """

    choice = np.random.choice(3)
    if choice == 0:
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:
        return load_image(data_dir, right), steering_angle - 0.2
    return load_image(data_dir, center), steering_angle


def random_flip(image, steering_angle):
    """
    Flip the input image horizontally and perform a steering angle adjustment at random.

    :param image: input frame
    :param steering_angle: steering angle related to the input frame
    :return: flipped input frame and adjusted steering angle
    """

    if np.random.rand() < 1:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def random_shift(image, steering_angle, range_x, range_y):
    """
    Shift (translate) the input image and perform a steering angle adjustment.

    :param image: input frame
    :param steering_angle: steering angle related to the input frame
    :param range_x: horizontal shift range
    :param range_y: vertical shift range
    :return: shifted version of the input frame and steering angle
    """

    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def random_shadow(image):
    """
    Add shadow to the input frame.

    :param image: input frame
    :return: shaded input frame
    """

    bright_factor = 0.3
    x = random.randint(0, image.shape[1])
    y = random.randint(0, image.shape[0])
    width = random.randint(image.shape[1], image.shape[1])
    if x + width > image.shape[1]:
        x = image.shape[1] - x
    height = random.randint(image.shape[0], image.shape[0])
    if y + height > image.shape[0]:
        y = image.shape[0] - y
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image[y:y + height, x:x + width, 2] = image[y:y + height, x:x + width, 2] * bright_factor
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)


def random_brightness(image):
    """
    Alter the brightness of the input image.

    :param image: input frame
    :return: altered input image
    """

    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + (np.random.rand() - 0.5)
    hsv[:, :, 2] = hsv[:, :, 2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augment(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
    """
    Generate an augmented image and adjust the associated steering angle.

    :param data_dir: images directory
    :param center: center view image
    :param left: left view image
    :param right: right view image
    :param steering_angle: the steering angle related to the input frame
    :param range_x: horizontal translation range
    :param range_y: vertical translation range
    :return: modified version of the input frame and steering angle
    """

    image, steering_angle = random_adjust(data_dir, center, left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_shift(image, steering_angle, range_x, range_y)
    image = random_shadow(image)
    image = random_brightness(image)
    return image, steering_angle


def batcher(dt_dir, image_paths, steering_angles, batch_size, training_flag, to_csv=False):
    """
    Generate batches of training images from the given image paths and their associated steering angles.

    :param dt_dir: the directory where the images are
    :param image_paths: paths to the input images
    :param steering_angles: the steering angles related to the input frames
    :param batch_size: the batch size used to train the model
    :param training_flag: a boolean flag to determine whether we are in training or validation mode
    :param to_csv: a boolean flag to decide if we are augmenting data and saving the angles in a CSV file
    :return: batches of images and steering angles
    """

    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    permuted = np.random.permutation(image_paths.shape[0])  # Global seed set

    count = 0
    while True:
        batch = permuted[count:count + batch_size]
        curr_bs = batch.shape[0]

        if image_paths.shape[0] <= count and to_csv:
            break
        # if batch.size == 0:
        #     break
        assert batch.size != 0
        for idx, index in enumerate(batch):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            if training_flag and np.random.rand() < 0.6:  # 60% probability of augmenting the data
                image, steering_angle = augment(dt_dir, center, left, right, steering_angle)
            else:
                image = load_image(dt_dir, center)

            # Populate the image and steering angle arrays
            images[idx] = preprocess(image)
            steers[idx] = steering_angle

        count += batch_size
        if curr_bs < batch_size:
            count = 0  # Reset the counter
        yield images[:curr_bs], steers[:curr_bs]


def train_model(mdl, x_train, x_valid, y_train, y_valid, model_name, cps_path='checkpoints/model-{val_loss:03f}.h5'):
    """
    Train a model.

    :param mdl: Keras sequential or functional model
    :param x_train: training procedure input data
    :param x_valid: validation procedure input data
    :param y_train: training procedure output (label) data
    :param y_valid: validation procedure output (label data
    :param model_name: name of the model
    :param cps_path: checkpoints path where the trained models are stored
    :return: None
    """

    # Checkpoint callback used to save the trained models
    checkpoint = ModelCheckpoint(cps_path,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='auto')

    # Compile the model
    mdl.compile(loss='mse',
                optimizer=Adam(lr=LR))

    # Train the model
    history = mdl.fit(batcher(TRAIN_DATA_DIR, x_train, y_train, B_SIZE, True),
                      steps_per_epoch=np.ceil(len(x_train) / B_SIZE),
                      epochs=NB_EPOCHS,
                      validation_data=batcher(TRAIN_DATA_DIR, x_valid, y_valid, B_SIZE, False),
                      validation_steps=np.ceil(len(x_valid) / B_SIZE),
                      callbacks=[checkpoint],
                      verbose=1)

    # Plot the training and validation losses
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_name} Model Loss (learning rate: {LR})')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'val_loss'], loc='upper left')
    plt.show()
