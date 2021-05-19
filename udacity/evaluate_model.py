"""
Python program employed in testing the various developed architecture pretrained models.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable debugging logs

import random
import numpy as np
import tensorflow as tf
import argparse
from kerasncp.tf import LTCCell
from tensorflow.python.keras.models import load_model
from utils import batcher, load_data
from constants import B_SIZE, TEST_DATA_DIR, TEST_LABELS_FL_DIR
import pandas as pd
import matplotlib.pyplot as plt

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Model evaluation program.')

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

parser.add_argument(
    '-path',
    action='store',
    help='enable the user to define a pretrained model path'
)

# Parse the arguments
args = parser.parse_args()

if not args.unseeded:
    print('\nSeeded model evaluation using GPU in progress...\n')

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
    print('\nUnseeded model evaluation using GPU in progress...\n')

    # https://www.tensorflow.org/guide/gpu
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

# Load the jungle track data
data = load_data(TEST_LABELS_FL_DIR)

# Initialise the evaluation data generator
test_gen = batcher(TEST_DATA_DIR, *data,
                   batch_size=B_SIZE,
                   training_flag=False)


def evaluate_model(mdl_name, mdl, generator):
    """
    Evaluate the provided model architecture and print out the results.

    :param mdl_name: model name
    :param mdl: loaded pretrained model
    :param generator: evaluation data generator
    :return: None
    """
    print(f'\n{mdl_name} model evaluation:')
    res = mdl.evaluate(generator,
                       steps=np.ceil(len(data[0]) / B_SIZE),
                       verbose=1)
    print(f'Test loss (MSE): {res}')


def plot_prediction(mdl_name, mdl, generator):
    """
    Plot a single batch predictions as opposed to the actual steering angle values.

    :param mdl_name: model name
    :param mdl: loaded pretrained model
    :param generator: evaluation data generator
    :return: None
    """

    for batch in generator:
        pred = mdl.predict(batch[0])

        y_actual = batch[1]
        y_pred = pred
        plt.figure(figsize=(11, 6))
        plt.plot(y_actual, linestyle='solid', color='r')
        plt.plot(y_pred, linestyle='dashed', color='b')

        plt.legend(['Actual', 'Predicted'], loc='best', prop={'size': 14})
        plt.title(f'{mdl_name} Model Actual vs Predicted Angles', weight='bold', fontsize=16)
        plt.ylabel('Angle (scaled)', weight='bold', fontsize=14)
        plt.xlabel('Sample no.', weight='bold', fontsize=14)
        plt.xticks(weight='bold', fontsize=12, rotation=45)
        plt.yticks(weight='bold', fontsize=12)
        plt.grid(color='y', linewidth='0.5')
        plt.show()
        return


def eval_predict(mdl_path, mdl_name, gen, custom_lyr=None):
    """
    Evaluate the model provided and plot its predictions against
    the actual steering angle values.

    :param mdl_path: path to the pretrained model checkpoint
    :param mdl_name: name of the model
    :param gen: evaluation data generator
    :param custom_lyr: custom layer (applied for the hybrid models)
    :return: None
    """
    mdl = load_model(mdl_path, custom_objects={'CustomLayer': custom_lyr})
    evaluate_model(mdl_name, mdl, gen)
    plot_prediction(mdl_name, mdl, gen)


# Store pretrained architecture paths and names in a dictionary.
# This dictionary is utilised to get the necessary model from the parsed command line arguments.
arch_dict = {}

# Set the dictionary values to the seeded or unseeded pretrained version of the models.
# Consider the '-unseeded' command line argument.
if not args.unseeded:
    arch_dict = {'cnn': ['checkpoints/cnn/seeded/cnn_model-0.009425.h5', 'CNN'],
                 'ncp': ['checkpoints/cnn_ncp/seeded/cnn_ncp_model-0.009680.h5', 'CNN-NCP'],
                 'dncp': ['checkpoints/cnn_dncp/seeded/cnn_dncp_model-0.009806.h5', 'CNN-DNCP'],
                 'dncp2': ['checkpoints/cnn_dncp_v2/seeded/cnn_dncp_v2_model-0.009748.h5', 'CNN-DNCP v2'],
                 'dncp3': ['checkpoints/cnn_dncp_v3/seeded/cnn_dncp_v3_model-0.009826.h5', 'CNN-DNCP v3'],
                 'dncp4': ['checkpoints/cnn_dncp_v4/seeded/cnn_dncp_v4_model-0.010126.h5', 'CNN-DNCP v4']
                 }
else:
    # The v3 and v4 models lack unseeded checkpoints due to project time constraints.
    # Therefore their seeded variants are used instead.
    arch_dict = {'cnn': ['checkpoints/cnn/unseeded/cnn_model-0.008119.h5', 'CNN'],
                 'ncp': ['checkpoints/cnn_ncp/unseeded/cnn_ncp_model-0.008284.h5', 'CNN-NCP'],
                 'dncp': ['checkpoints/cnn_dncp/unseeded/cnn_dncp_model-0.008643.h5', 'CNN-DNCP'],
                 'dncp2': ['checkpoints/cnn_dncp_v2/unseeded/cnn_dncp_v2_model-0.008334.h5', 'CNN-DNCP v2'],
                 'dncp3': ['checkpoints/cnn_dncp_v3/seeded/cnn_dncp_v3_model-0.009826.h5', 'CNN-DNCP v3'],
                 'dncp4': ['checkpoints/cnn_dncp_v4/seeded/cnn_dncp_v4_model-0.010126.h5', 'CNN-DNCP v4']
                 }

try:
    if args.path:
        # Fetch the model and the model and plot names
        model_path, model_name = args.path, arch_dict[args.model][1]
    else:
        # Fetch the model and the model and plot names
        model_path, model_name = arch_dict[args.model]

    # Evaluate the given pretrained model type with the best MSE result the jungle track dataset (Udacity)
    eval_predict(model_path, model_name, test_gen, custom_lyr=LTCCell)
except OSError:
    print('Incorrect model path. Please try again!')
