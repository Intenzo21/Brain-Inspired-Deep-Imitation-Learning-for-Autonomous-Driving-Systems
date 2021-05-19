"""
Python program employed in training the various developed architecture models.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable debugging logs
import random
import numpy as np
import tensorflow as tf
import argparse
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from data_generator import DataGenerator
from glob import glob  # Finds all path names matching specified pattern
import matplotlib.pyplot as plt
from models import cnn_model, cnn_ncp_model, cnn_dncp_model, cnn_dncp_v2_model, cnn_dncp_v3_model, cnn_dncp_v4_model
from constants import TEST_SIZE, RANDOM_STATE, TRAIN_DATA_DIR, TRAIN_LABELS_FL_DIR, LR, VERBOSITY, NB_EPOCHS

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Model training program.')

# Fill the ArgumentParser object with information about program arguments.
parser.add_argument(
    'model',
    type=str,
    metavar='model',
    choices=['cnn', 'ncp', 'dncp', 'dncp2', 'dncp3', 'dncp4'],
    help='model type'
)

parser.add_argument(
    '-unseeded',
    action='store_true',
    help='leave the random script values unseeded'
)

# Parse the arguments
args = parser.parse_args()

if not args.unseeded:
    print('\nSeeded model training in progress...\n')

    # Use the same seeding value for all random number packages
    seed_value = 1337

    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value

    os.environ['PYTHONHASHSEED'] = str(seed_value)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # 2. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # 3. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)  # tensorflow 2.x
else:
    print('\nUnseeded model training using GPU in progress...\n')


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


# Get the HDF5 sunny camera and log files
cam_files = sorted(glob(TRAIN_DATA_DIR))
log_files = sorted(glob(TRAIN_LABELS_FL_DIR))

# Split the sunny comma.ai videos whose camera frame recordings have been shrunk to 80 x 120 (height x width).
# The dataset training to validation set ratio is 80/20.
X_train, X_val, y_train, y_val \
    = train_test_split(cam_files, log_files, test_size=TEST_SIZE, random_state=RANDOM_STATE)


# Instantiate the train, validation and test data generators
train_gen = DataGenerator(X_train, y_train)
val_gen = DataGenerator(X_val, y_val)

# Store architecture function references and model names in a dictionary.
# This dictionary is utilised to get the necessary model from the parsed command line arguments.
arch_dict = {'cnn': [cnn_model, 'cnn', 'CNN'],
             'ncp': [cnn_ncp_model, 'cnn_ncp', 'CNN-NCP'],
             'dncp': [cnn_dncp_model, 'cnn_dncp', 'CNN-DNCP'],
             'dncp2': [cnn_dncp_v2_model, 'cnn_dncp_v2', 'CNN-DNCP v2'],
             'dncp3': [cnn_dncp_v3_model, 'cnn_dncp_v3', 'CNN-DNCP v3'],
             'dncp4': [cnn_dncp_v4_model, 'cnn_dncp_v4', 'CNN-DNCP v4']
             }

model, model_name, model_name_plot = arch_dict[args.model]

# Initialise the optimiser
optimizer = Adam(learning_rate=LR)

# Get the Keras sequential/functional model by using its reference
model = model()

# Compile the model
model.compile(loss='mse',
              optimizer=optimizer,
              )

# The path of the model checkpoints
cps_path = ''

# Organise into seeded and unseeded checkpoint folders
if not args.unseeded:
    cps_path = f'checkpoints/{model_name}/seeded/{model_name}_model' + '-{val_loss:03f}.h5'
else:
    cps_path = f'checkpoints/{model_name}/unseeded/{model_name}_model' + '-{val_loss:03f}.h5'

# Create a Keras 'ModelCheckpoint' callback to save the best model
checkpoint = ModelCheckpoint(cps_path,
                             monitor='val_loss',
                             verbose=VERBOSITY,
                             save_best_only=True,
                             mode='auto')

# Start training the model
history = model.fit(train_gen,
                    epochs=NB_EPOCHS,
                    verbose=VERBOSITY,
                    validation_data=val_gen,
                    callbacks=[checkpoint]
                    )

# Plot the training and validation losses
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title(f'{model_name_plot} Model Loss (learning rate: {LR})')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val_loss'], loc='upper left')
plt.show()

