# import os
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable debugging logs
seed_value = 1337

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random

random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np

np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf

tf.random.set_seed(seed_value)  # tensorflow 2.x

from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from keras_lr_finder import LRFinder
from keras import regularizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from data_generator import DataGenerator
from glob import glob  # Finds all path names matching specified pattern
import matplotlib.pyplot as plt
from models import other_model, NVIDIA_model, stacked_model, stacked_dual_model, stacked_dual_model_v2

# https://www.tensorflow.org/guide/gpu
# In our case we want the processes to only grow the memory usage as is needed
# by them. To control this we employ a TensorFlow method. To turn on memory growth
# we call the 'tf.config.experimental.set_memory_growth' method, which attempts to
# allocate only as much GPU memory as needed for the runtime allocations:
# it starts out allocating very little memory, and as the program gets run and more GPU memory is
# needed, we extend the GPU memory region allocated to the TensorFlow process.

gpus = tf.config.experimental.list_physical_devices('GPU')  # Confirm that TensorFlow is using the GPU.

# To turn on memory growth for a specific GPU, we use the following code prior to allocating
# any tensors or executing any ops.
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# class Trainer:
#
#     def __init__(self, ds_frames, ds_angles):
#         self.ds_frames = ds_frames
#         self.ds_angles = ds_angles
#         self.X_train, self.X_val, self.y_train, self.y_val = self.split_data()
#
#     def split_data(self, test_size=0.2):
#         return train_test_split(self.ds_frames, self.ds_angles, test_size=test_size, random_state=42)
#
#     def train(self):
#         pass


# Wrong aligning of logs (steering angles) with frames
# Introduce shuffling of frames and angles before batching
# CNN gives better training loss and converges faster than the stacked model
# Should I use more data in that situation or that is enough?
# What lr to use among the ones I sent?
# Maybe the stacked model gives better evaluation loss on completely unseen (different) data.
# Here the two models will be tested on 20% of the shrunk dataset. This means that in the test
# set there will be data from the recordings used for training and evaluation. Therefore, it will not be
# 'completely' unseen data.
# Nvidia around 1.25 times better validation loss

# Model configuration
batch_size = 64
no_epochs = 10
learning_rate = 5e-06  # Since trains the model to the best val_loss in 10 epochs (model starts overfitting)

# Adam combines the best properties of the AdaGrad and RMSProp
# algorithms to provide an optimization algorithm that can handle
# sparse gradients on noisy problems. (AdaDelta is basically an
# extension of AdaGrad that seeks to reduce its aggressive,
# monotonically decreasing learning rate.)
# https://ruder.io/optimizing-gradient-descent/index.html#adadelta
# Also Nadam too slow.
optimizer = Adam(learning_rate=learning_rate)
verbosity = 1

# model = NVIDIA_model()  # time=17s 0020 lr=1e-05 | 0022 lr=5e-06
model = stacked_model()  # time=38s 0021 lr=1e-05 | 0021 lr=5e-06
# model = stacked_dual_model()
# model = stacked_dual_model_v2()
MODEL_NAME = 'stacked'

# Compile the model
model.compile(loss='mse',
              optimizer=optimizer,
              )

cam_files = sorted(glob("./dataset/64_batched_sunny_same/frames/*.npy"))
log_files = sorted(glob("./dataset/64_batched_sunny_same/angles/*.npy"))

# Split the whole comma.ai dataset whose camera frame recordings have been shrunk to 80x80 (height x width).
# The dataset train, test, validation percentages are 60%, 20% and 20% respectively.
# the 75% of the dataset is split into training and test sets (37.5% each)
X_train, X_val, y_train, y_val \
    = train_test_split(cam_files, log_files, test_size=0.2, random_state=42)

# for i in range(20):
#     print(X_train[i], y_train[i])
#     input()

# # 7.5 % val data and 30 % train data
# X_train, X_val, y_train, y_val \
#     = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.8 * 0.25 = 0.2

# _, X_test, _, y_test = \
#     train_test_split(X_test, y_test, test_size=0.2, random_state=42)  # 0.8 * 0.25 = 0.2

# Instantiate the train, validation and test data generators
train_gen = DataGenerator(X_train, y_train)
val_gen = DataGenerator(X_val, y_val)
# test_gen = DataGenerator(X_test, y_test)

# lr_finder = LRFinder(model)
#
# # Train a model with batch size 512 for 5 epochs
# # with learning rate growing exponentially from 0.0001 to 1
# lr_finder.find_generator(train_gen, start_lr=learning_rate, end_lr=1, epochs=10)
# lr_finder.plot_loss()
# input()

# Create a Keras 'ModelCheckpoint' callback to save the best model
cps_path = f'checkpoints/sunny_dataset_same/{MODEL_NAME}_{learning_rate}/new/{MODEL_NAME}_model-' + '{val_loss:03f}.h5'
checkpoint = ModelCheckpoint(cps_path,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='auto')

# reduce_lr = ReduceLROnPlateau(monitor='val_loss',
#                               factor=10,
#                               patience=1,
#                               min_lr=0.1e-7,
#                               verbose=1)

# Start training the model
history = model.fit(train_gen,
                    epochs=10,
                    verbose=verbosity,
                    validation_data=val_gen,
                    callbacks=[checkpoint]
                    )

# Plot the training and validation losses
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title(f'Stacked (CNN+NCP) Model Loss (learning rate: {learning_rate})')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val_loss'], loc='upper left')
plt.show()
