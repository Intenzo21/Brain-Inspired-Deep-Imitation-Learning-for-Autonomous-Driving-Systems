import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.transform import resize
import dask.array as da
from utils import scale_data, save_data, npy_chunks, standardize_data, plot_data
from numpy import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle


# TODO: maybe mention trying out the below methods which were later omitted:

# Adopt Dask library in case the dataset file is too large for a NumPy array to be initialized
# if cam_fl.shape[0] <= 55000:
#     cam_fl = cam_fl[start_idx:end_idx]
# else:
#     cam_fl = da.from_array(cam_fl)[start_idx:end_idx]


# if self.shrink_size:
#     # Reduce the dataset to 50% to overcome the NumPy array memory allocation issue
#     # Also focus on the parts of the video recordings where more steering is performed
#     shrink_idx = int(len(cam_pointers) * self.shrink_size)
#     start_idx = int(cam_pointers[shrink_idx])  # Skip the camera frames to the highway
#     # part of the video. Car takes time
#     # to start therefore no movement at the beginning => can be omitted
#     end_idx = int(cam_pointers[-shrink_idx])

#     # Make room for new data
#     self.all_frames.resize((abs_end, 3, 80, 80))
#     self.str_angles.resize(abs_end)
#
#     # Insert data into arrays
#     self.all_frames[abs_start:abs_end] = cam_80
#     starts = range(start_idx, end_idx)
#     # print(starts[0], starts[-1], len(starts))
#     # input()
#     self.str_angles[abs_start:abs_end] = log_fl['steering_angle'][...][starts]  # Get multiple index values
#     print(self.str_angles)
#
# # Save the shrunk image dataset to a zip file
# self.str_angles = scale_data(self.str_angles)
# save_data(self.file_name,
#           all_frames=self.all_frames,
#           str_angles=self.str_angles,
#           fl_starts=self.fl_starts)

# scaler = MinMaxScaler((-1, 1))
# str_angles = np.reshape(str_angles, (-1, 1))
# str_angles = scaler.fit_transform(str_angles)
# str_angles = standardize_data(str_angles)


class DataPreprocessor:

    def __init__(self, cam_files, log_files, folder_name, batch_size=32, ds_min=None, ds_max=None, shrink_size=None,
                 resize_dims=(80, 80), crop_size=(None, None)):
        self.cam_files = cam_files
        self.log_files = log_files
        self.shrink_size = shrink_size / 2 if shrink_size else None
        self.resize_dims = resize_dims
        self.crop_size = crop_size
        self.all_frames = np.zeros((1, 3, *self.resize_dims), dtype=np.uint8)
        self.str_angles = np.zeros(1)
        self.folder_name = folder_name
        self.batch_size = batch_size
        self.ds_min = ds_min
        self.ds_max = ds_max

        # List to store the start indexes of each video file
        self.fl_starts = [0]  # type: list

    def run(self):
        chunk_start = 0

        for c, l in zip(self.cam_files, self.log_files):
            print(c, l)
            cam_fl = h5py.File(c, 'r')['X']  # Open the camera file for reading
            n_frames = cam_fl.shape[0]  # Number of camera frames

            log_fl = h5py.File(l, 'r')  # Open the log file for reading
            cam_pointers = log_fl['cam1_ptr']  # List of camera pointers to be used
            # in aligning the measurements with the frames
            starts = np.zeros(n_frames, dtype=np.uint32)
            start_idx = 0
            end_idx = cam_fl.shape[0]

            print(cam_pointers[:100], cam_fl.shape[0], cam_pointers[-4])

            curr = 1
            # Best way yet since at times there are 6 logs for a single frame.
            # Therefore cannot use range with 5 steps...
            for idx, p in tqdm(enumerate(cam_pointers)):  # 263575
                if int(p) == curr:
                    starts[curr] = idx
                    curr += 1
            starts[0] = starts[1] - 5
            # starts = starts[::5]  # Take 20%

            # # Keep track of the data amount
            # abs_start = self.fl_starts[-1]
            # abs_end = abs_start + n_frames
            # self.fl_starts.append(abs_end)

            print(n_frames, starts.shape[0])
            # Shrink down cam frames to 80x80
            cam_80 = np.zeros((n_frames, 3, 80, 120), dtype=np.uint8)

            count = 0
            for i, fr in tqdm(enumerate(cam_fl)):
                # if i % 5 == 0:
                cam_80[i] = resize(fr[:, :, self.crop_size[0]:self.crop_size[1]].astype('uint8'),
                                   (3, 80, 120), preserve_range=True)  #
                # count += 1
            # print(log_fl['steering_angle'][...][starts[i]])
            # plt.imshow(cam_80[i].transpose(1, 2, 0))
            # plt.show()
            # input()

            cam_80, starts = shuffle(cam_80, starts, random_state=1337)

            npy_chunks(cam_80, self.folder_name, 'frames', self.batch_size, chunk_start)

            str_angles = log_fl['steering_angle'][...][starts]  # Get multiple index values
            str_angles = scale_data(str_angles, self.ds_min, self.ds_max)

            for i in range(1):
                plt.imshow(cam_fl[int(cam_pointers[starts[i]])].transpose(1, 2, 0))
                plt.show()
                plt.imshow(cam_80[i].transpose(1, 2, 0))
                plt.show()

            plot_data(str_angles, 'Steering angles', 'Frame', 'Steering angle (scaled)')
            chunk_start += npy_chunks(str_angles, self.folder_name, 'angles', self.batch_size, chunk_start)




