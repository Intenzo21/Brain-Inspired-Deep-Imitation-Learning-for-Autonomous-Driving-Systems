import argparse
from glob import glob
import matplotlib.pyplot as plt
import numpy as np

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Data distribution plotting program.')

# Fill the ArgumentParser object with information about program arguments.
parser.add_argument(
    'rec_type',
    type=str,
    metavar='rec_type',
    choices=['sunny', 'cloudy', 'night'],
    help='recording type'
)

# Parse the arguments
args = parser.parse_args()


def plot_distribution(data, bins):
    """
    Plot the distribution of the given data.

    :param data: data list or array
    :param bins: number of columns in the plot
    :return: None
    """
    plt.hist(data, color='blue', edgecolor='black',
             bins=bins)

    plt.title('Histogram of Steering Angles')
    plt.xlabel('Steering angle')
    plt.ylabel('Count')
    plt.show()


# Fetch the NumPy filenames of the batched dataset angles
log_files = sorted(glob(f"./dataset/64_batched_{args.rec_type}/angles/*.npy"))

# Create a list containing all of the specified video type steering angle values
angles = [angle for i in range(len(log_files)) for angle in np.load(log_files[i])]

# Plot the data distribution of the angles
plot_distribution(angles, 20)
