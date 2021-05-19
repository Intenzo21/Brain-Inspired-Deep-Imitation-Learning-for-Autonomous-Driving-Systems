"""
DataPreprocessor class implementation.
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.transform import resize

from constants import B_SIZE, RESIZE_DIMS, CROP_SIZE
from utils import scale_data, save_data, npy_chunks, standardise_data, plot_data
from sklearn.utils import shuffle


class DataPreprocessor:
    """
    Class for creating data preprocessor instances.
    """

    def __init__(self, cam_files, log_files, folder_name, batch_size=B_SIZE, ds_min=None, ds_max=None, shrink_size=None,
                 resize_dims=RESIZE_DIMS, crop_size=CROP_SIZE):
        """
        Set the data preprocessor instance attributes.

        :param cam_files: input camera frame recordings
        :param log_files: measurement log files
        :param folder_name: name of the folder where the preprocessed data will be stored
        :param batch_size: preprocessing batch size (set to 64 by default)
        :param ds_min: minimum steering angle value extracted from the whole comma.ai dataset (for scaling purposes)
        :param ds_max: maximum steering angle value extracted from the whole comma.ai dataset (for scaling purposes)
        :param shrink_size: value to shrink the start and end of the dataset recordings
        :param resize_dims: dimensions of the resized camera frames
        :param crop_size: height x width cropping size
        """
        self.cam_files = cam_files
        self.log_files = log_files
        self.shrink_size = shrink_size / 2 if shrink_size else None
        self.resize_dims = resize_dims
        self.crop_size = crop_size
        self.str_angles = np.zeros(1)
        self.folder_name = folder_name
        self.batch_size = batch_size
        self.ds_min = ds_min
        self.ds_max = ds_max

    def run(self):
        """
        Run the data preprocessor instance.

        :return: None
        """

        # Iterate over the recordings and their corresponding log files
        chunk_start = 0
        for c, l in zip(self.cam_files, self.log_files):

            # Open the camera file for reading
            cam_fl = h5py.File(c, 'r')['X']

            # Number of camera frames
            n_frames = cam_fl.shape[0]

            # Open the log file for reading
            log_fl = h5py.File(l, 'r')

            # List of camera pointers to be used
            cam_pointers = log_fl['cam1_ptr']

            # Variable used in aligning the measurements with the frames
            starts = np.zeros(n_frames, dtype=np.uint32)

            curr = 1  # Counter variable
            # Best way yet since at times there are 6 logs for a single frame.
            # Therefore cannot use range with 5 steps...
            for idx, p in tqdm(enumerate(cam_pointers)):  # 263575
                if int(p) == curr:
                    starts[curr] = idx
                    curr += 1
            starts[0] = starts[1] - 5

            # Shrink down cam frames to 80 x 120 by resizing
            cam_80 = np.zeros((n_frames, 3, *self.resize_dims), dtype=np.uint8)
            for i, fr in tqdm(enumerate(cam_fl)):
                cam_80[i] = resize(fr[:, :, self.crop_size[0]:self.crop_size[1]].astype('uint8'),
                                   (3, *self.resize_dims), preserve_range=True)

            # Shuffle the resized camera frames and the steering angle start indexes
            cam_80, starts = shuffle(cam_80, starts, random_state=1337)

            # Save the shuffled and resized camera frames as indexed, batch-sized NumPy files
            npy_chunks(cam_80, self.folder_name, 'frames', self.batch_size, chunk_start)

            # Get multiple index values of the steering angles and scale them
            str_angles = log_fl['steering_angle'][...][starts]
            str_angles = scale_data(str_angles, self.ds_min, self.ds_max)

            # Code for visualising the process (testing)
            # for i in range(1):
            #     plt.imshow(cam_fl[int(cam_pointers[starts[i]])].transpose(1, 2, 0))
            #     plt.show()
            #     plt.imshow(cam_80[i].transpose(1, 2, 0))
            #     plt.show()
            #
            # plot_data(str_angles, 'Steering angles', 'Frame no.', 'Steering angle (scaled)')

            # Save the shuffled and scaled steering angle values as indexed, batch-sized NumPy files
            chunk_start += npy_chunks(str_angles, self.folder_name, 'angles', self.batch_size, chunk_start)
