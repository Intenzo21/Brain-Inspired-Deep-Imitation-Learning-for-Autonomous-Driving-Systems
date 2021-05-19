"""
Python program employed in evaluating the various
developed architecture pretrained models. Along with this
by running the script a prediction plot is created.
"""

# Importing Python libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable debugging logs

import random
import numpy as np
import tensorflow as tf
import argparse
from kerasncp.tf import LTCCell
from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt
from glob import glob
from data_generator import DataGenerator
from sklearn.utils import shuffle

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
    'gen',
    type=str,
    metavar='gen',
    choices=['cloudy', 'night'],
    help='evaluation data generator type'
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


def eval_model(model, mdl_name, gen):
    """
    Evaluate model and print the test loss

    :param tf.keras.Model model:
    :param string mdl_name: name of the model
    :param DataGenerator gen: DataGenerator instance (test data generator in this case)
    :return: None
    """

    print(f'\n{mdl_name} model evaluation:')
    res = model.evaluate(gen)
    print(f'Test loss (MSE): {res}')


def plot_angles(mdl_name, actual, predicted):
    """
    Plot the actual vs predicted steering angle values.

    :param mdl_name: name of the model
    :param actual: actual steering angle values data
    :param predicted: predicted steering angle values data
    :return: None
    """

    plt.figure(figsize=(11, 6))
    plt.plot(actual, linestyle='solid', color='r')
    plt.plot(predicted, linestyle='dashed', color='b')
    plt.legend(['Actual', 'Predicted'], loc='best', prop={'size': 14})
    plt.title(f'{mdl_name} Actual vs Predicted Steering Angle Values', weight='bold', fontsize=16)
    plt.ylabel('Angle (scaled)', weight='bold', fontsize=14)
    plt.xlabel('Batch Sample no.', weight='bold', fontsize=14)
    plt.xticks(weight='bold', fontsize=12, rotation=45)
    plt.yticks(weight='bold', fontsize=12)
    plt.grid(color='y', linewidth='0.5')
    plt.show()


def eval_predict(mdl_path, mdl_name, gen, custom_lyr=LTCCell):
    """
    Evaluate the model and plot its angle predictions agains the actual ones.

    :param mdl_path: path to the model
    :param mdl_name: name of the model
    :param gen: evaluation data generator
    :param custom_lyr: custom layer (hybrid models)
    :return: None
    """
    # Load and evaluate the CNN checkpoint model
    model = load_model(mdl_path, custom_objects={'CustomLayer': custom_lyr})
    model.summary()
    eval_model(model, mdl_name, gen)

    # Predict angles for the first 64 frames
    # Predict steering angle output values using a single batch from the test data generator
    y_pred = model(gen[0][0])  # Predicted steering angle output values
    y_actual = gen[0][1]  # Actual (expected) steering angle output values
    plot_angles(model_name, y_actual, y_pred)


# Fetch the batched dataset frame and angle NumPy filenames
# Here we adopt the cloudy dataset recording files to evaluate
# the checkpoint models (best loss).
eval_gen_type = args.gen
cam_files = sorted(glob(f"./dataset/64_batched_{eval_gen_type}/frames/*.npy"))
log_files = sorted(glob(f"./dataset/64_batched_{eval_gen_type}/angles/*.npy"))

# Shuffle the frame and angle filenames
cam_files, log_files = shuffle(cam_files, log_files, random_state=42)

# Initialize the evaluation data generator
eval_gen = DataGenerator(cam_files, log_files)

# Store pretrained architecture paths and names in a dictionary.
# This dictionary is utilised to get the necessary model from the parsed command line arguments.
arch_dict = {}

# Set the dictionary values to the seeded or unseeded pretrained version of the models.
# Consider the '-unseeded' command line argument.
if not args.unseeded:
    arch_dict = {'cnn': ['checkpoints/cnn/seeded/cnn_model-0.002091.h5', 'CNN'],
                 'ncp': ['checkpoints/cnn_ncp/seeded/cnn_ncp_model-0.002154.h5', 'CNN-NCP'],
                 'dncp': ['checkpoints/cnn_dncp/seeded/cnn_dncp_model-0.007434.h5', 'CNN-DNCP'],
                 'dncp2': ['checkpoints/cnn_dncp_v2/seeded/cnn_dncp_v2_model-0.002150.h5', 'CNN-DNCP v2'],
                 'dncp3': ['checkpoints/cnn_dncp_v3/seeded/cnn_dncp_v3_model-0.002125.h5', 'CNN-DNCP v3'],
                 'dncp4': ['checkpoints/cnn_dncp_v4/seeded/cnn_dncp_v4_model-0.002156.h5', 'CNN-DNCP v4']
                 }
else:
    # The original CNN-DNCP, v3 and v4 models lack unseeded checkpoints due to project time constraints.
    # Therefore their seeded variants are used instead.
    arch_dict = {'cnn': ['checkpoints/cnn/unseeded/cnn_model-0.002053.h5', 'CNN'],
                 'ncp': ['checkpoints/cnn_ncp/unseeded/cnn_ncp_model-0.002101.h5', 'CNN-NCP'],
                 'dncp': ['checkpoints/cnn_dncp/seeded/cnn_dncp_model-0.007434.h5', 'CNN-DNCP'],
                 'dncp2': ['checkpoints/cnn_dncp_v2/unseeded/cnn_dncp_v2_model-0.002107.h5', 'CNN-DNCP v2'],
                 'dncp3': ['checkpoints/cnn_dncp_v3/seeded/cnn_dncp_v3_model-0.002125.h5', 'CNN-DNCP v3'],
                 'dncp4': ['checkpoints/cnn_dncp_v4/seeded/cnn_dncp_v4_model-0.002156.h5', 'CNN-DNCP v4']
                 }

try:
    if args.path:
        # Fetch the model and the model and plot names
        model_path, model_name = args.path, arch_dict[args.model][1]
    else:
        # Fetch the model and the model and plot names
        model_path, model_name = arch_dict[args.model]

    # Run the model evaluator and prediction plot function
    eval_predict(model_path, model_name, eval_gen)
except OSError:
    print('Incorrect model path. Please try again!')