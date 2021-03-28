from glob import glob

import h5py
from tqdm import tqdm

from data_preprocessor import DataPreprocessor

from utils import npy_chunks

import numpy as np

# Get the camera recording and log file names (.h5 formatted)
# Sort them just in case
s_cam_fls = sorted(glob("comma-dataset/cam/sunny/*.h5"))
c_cam_fls = sorted(glob("comma-dataset/cam/cloudy/*.h5"))
n_cam_fls = sorted(glob("comma-dataset/cam/night/*.h5"))
s_log_fls = sorted(glob("comma-dataset/log/sunny/*.h5"))
c_log_fls = sorted(glob("comma-dataset/log/cloudy/*.h5"))
n_log_fls = sorted(glob("comma-dataset/log/night/*.h5"))

# all_cams = s_cam_fls + c_cam_fls + n_cam_fls
# n_frames = 0
# for idx, l in tqdm(enumerate(s_cam_fls)):
#     print(idx)
#     cam_fl = h5py.File(l, 'r')['X']  # Open the log file for reading
#     n_frames += cam_fl.shape[0]
#
# # 153043
# # 252
# # ~ 0.165%
# print(n_frames)
# input()


# Extract the minimum and maximum steering angle values
# by going through the whole dataset. This is needed for
# the scaling pre-processing (data normalization) step
# of the steering angles (within the range of 0 and 1)
max_vals, min_vals = ([] for i in range(2))
all_logs = s_log_fls + c_log_fls + n_log_fls
for l in tqdm(all_logs):
    log_fl = h5py.File(l, 'r')  # Open the log file for reading
    max_vals.append(max(log_fl['steering_angle'][...]))
    min_vals.append(min(log_fl['steering_angle'][...]))

ds_max = max(max_vals)
ds_min = min(min_vals)
print(ds_min, ds_max)

s_folder_name = 'sunny'
c_folder_name = 'cloudy'
n_folder_name = 'night'
batch_size = 64

# Run the data preprocessor
data_preprocessor = DataPreprocessor(n_cam_fls, n_log_fls, n_folder_name, batch_size=batch_size,
                                     ds_min=ds_min, ds_max=ds_max, crop_size=(40, -40))
data_preprocessor.run()
