"""
Script for plotting steering angle distribution.
"""
import argparse

import pandas as pd
import matplotlib.pyplot as plt
from utils import batcher, load_data
from constants import TRAIN_DATA_DIR, TRAIN_LABELS_FL_DIR, TEST_LABELS_FL_DIR, TEST_DATA_DIR
import numpy as np
from tqdm import tqdm

# Create an ArgumentParser object

parser = argparse.ArgumentParser(description='Data distribution plotting program.')

sp = parser.add_subparsers()
# Fill the ArgumentParser object with information about program arguments.
path_parser = sp.add_parser(
    'csv_path',
    help='path to the CSV file'
)

creator_parser = sp.add_parser(
    'create_csv',
    help='enable the user to create a CSV with either augmented lakeside or jungle steering angles'
)

path_parser.add_argument(
    'csv_path',
    type=str,
    default=None,
    help='path to the CSV file'
)

creator_parser.add_argument('create_csv',
                            type=str,
                            choices=['lakeside', 'jungle'],
                            default=None,
                            help='enable the user to create a CSV with either augmented lakeside or jungle steering '
                                 'angles '
                            )
# Parse the arguments
args = parser.parse_args()


def plot_distribution(dt, bins):
    """
    Plot the given data distribution.

    :param dt: data
    :param bins: number of columns in the plot
    :return: None
    """

    # Matplotlib histogram
    plt.hist(dt, color='blue', edgecolor='black',
             bins=bins)

    plt.title('Histogram of Steering Angles')
    plt.xlabel('Steering Angle')
    plt.ylabel('Count')
    plt.show()


# Load the original steering angles
# driving_logs = pd.read_csv('data/lakeside_map/driving_log.csv')
if hasattr(args, 'create_csv'):
    # Create a CSV file with the augmented steering angles
    # to be used for plotting the data distribution.

    # Dictionary to get the data path from the command line argument
    data_dict = {'lakeside': [TRAIN_LABELS_FL_DIR, TRAIN_DATA_DIR],
                 'jungle': [TEST_LABELS_FL_DIR, TEST_DATA_DIR],
                 }

    # Fetch the data path and load
    labels, dt = data_dict[args.create_csv]
    data = load_data(labels)

    # Use a generator to create an array of steering angle values of the desired track data
    generator = batcher(dt, *data, batch_size=1, training_flag=True, to_csv=True)
    batches = np.array([angle[0] for _, angle in tqdm(generator)])

    # Save the steering angles to a CSV file
    np.savetxt(f'{args.create_csv}_angles_augmented.csv', batches, fmt='%.5f', delimiter=',', header='angles')
else:
    try:
        # Load the augmented steering angles
        angles = pd.read_csv(args.csv_path)

        # Plot the data distribution
        plot_distribution(angles['angles'], 20)
    except OSError:
        print('Incorrect model path. Please try again!')
