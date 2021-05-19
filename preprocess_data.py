"""
Script which creates and runs a DataPreprocessor class instance.
"""

import argparse
from glob import glob
import h5py
from tqdm import tqdm
from data_preprocessor import DataPreprocessor

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Data preprocessing program.')

# Fill the ArgumentParser object with information about program arguments.
parser.add_argument(
    'rec_type',
    type=str,
    metavar='rec_type',
    choices=['sunny', 'cloudy', 'night'],
    help='recording type'
)

parser.add_argument(
    'batch_size',
    type=int,
    metavar='batch_size',
    nargs='?',
    choices=[32, 64, 128, 256, 512],
    default=64,
    help='preprocessing batch size'
)

# Parse the arguments
args = parser.parse_args()

# Get the camera recording and log file names (.h5 formatted)
# Sort them just in case
s_cam_fls = sorted(glob("comma-dataset/cam/sunny/*.h5"))
c_cam_fls = sorted(glob("comma-dataset/cam/cloudy/*.h5"))
n_cam_fls = sorted(glob("comma-dataset/cam/night/*.h5"))
s_log_fls = sorted(glob("comma-dataset/log/sunny/*.h5"))
c_log_fls = sorted(glob("comma-dataset/log/cloudy/*.h5"))
n_log_fls = sorted(glob("comma-dataset/log/night/*.h5"))

# Extract the minimum and maximum steering angle values
# by going through the whole dataset. This is needed for
# the scaling pre-processing (data normalization) step
# of the steering angles (within the range of 0 and 1)
max_vals, min_vals = ([] for i in range(2))
all_logs = s_log_fls + c_log_fls + n_log_fls
for log in tqdm(all_logs):
    log_fl = h5py.File(log, 'r')  # Open the log file for reading
    max_vals.append(max(log_fl['steering_angle'][...]))
    min_vals.append(min(log_fl['steering_angle'][...]))

ds_max = max(max_vals)
ds_min = min(min_vals)

# Recording types dictionary
rec_dict = {'sunny': [s_cam_fls, s_log_fls],
            'cloudy': [c_cam_fls, c_log_fls],
            'night': [n_cam_fls, n_log_fls]
            }

# Get the argument values from the command line
cam_fls_name = args.rec_type
cam_fls, log_fls = rec_dict[cam_fls_name]
batch_size = args.batch_size

# Instantiate and run the data preprocessor
data_preprocessor = DataPreprocessor(cam_fls, log_fls, cam_fls_name, batch_size=batch_size,
                                     ds_min=ds_min, ds_max=ds_max)
data_preprocessor.run()
