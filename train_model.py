"""
Python program employed in training the various developed architecture models.
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable debugging logs
import random
import numpy as np
import tensorflow as tf
import argparse
from utils import load_data, train_model
from models import cnn_model, cnn_ncp_model, cnn_dncp_model, cnn_dncp_v2_model, cnn_dncp_v3_model, cnn_dncp_v4_model

from constants import TRAIN_LABELS_FL_DIR

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

    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value

    os.environ['PYTHONHASHSEED'] = str(seed_value)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # 2. Set the Python built-in pseudo-random generator at a fixed value
    random.seed(seed_value)

    # 3. Set the NumPy pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # 4. Set the Tensorflow pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)  # tensorflow 2.x
else:
    print('\nUnseeded model training using GPU in progress...\n')

    # Taken from https://www.tensorflow.org/guide/gpu
    # In our case we want the processes to grow the memory usage only when needed.
    # To control this we employ a TensorFlow method. To turn on memory growth
    # we call the 'tf.config.experimental.set_memory_growth' method, which attempts to
    # allocate only as much GPU memory as needed for the runtime allocations:
    # it starts out allocating very little memory, and as the program gets run and more GPU memory is
    # needed, we extend the GPU memory region allocated to the TensorFlow process.

    gpus = tf.config.experimental.list_physical_devices('GPU')
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

# Load the images data
data = load_data(TRAIN_LABELS_FL_DIR, 0.2)

# Store architecture function references and model names in a dictionary.
# This dictionary is utilised to get the necessary model from the parsed command line arguments.
arch_dict = {'cnn': [cnn_model, 'cnn', 'CNN'],
             'ncp': [cnn_ncp_model, 'cnn_ncp', 'CNN-NCP'],
             'dncp': [cnn_dncp_model, 'cnn_dncp', 'CNN-DNCP'],
             'dncp2': [cnn_dncp_v2_model, 'cnn_dncp_v2', 'CNN-DNCP v2'],
             'dncp3': [cnn_dncp_v3_model, 'cnn_dncp_v3', 'CNN-DNCP v3'],
             'dncp4': [cnn_dncp_v4_model, 'cnn_dncp_v4', 'CNN-DNCP v4']
             }

# Fetch the model and the model and plot names
model, model_name, model_name_plot = arch_dict[args.model]

# The path of the model checkpoints
cps_path = ''

# Organise into seeded and unseeded checkpoint folders
if not args.unseeded:
    cps_path = f'checkpoints/{model_name}/seeded/{model_name}_model' + '-{val_loss:03f}.h5'
else:
    cps_path = f'checkpoints/{model_name}/unseeded/{model_name}_model' + '-{val_loss:03f}.h5'

# Train the specified model
train_model(model(), *data, model_name=model_name_plot, cps_path=cps_path)
