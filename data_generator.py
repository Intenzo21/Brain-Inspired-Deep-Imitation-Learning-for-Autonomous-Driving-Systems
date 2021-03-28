import numpy as np
from keras.utils import Sequence
from glob import glob  # Finds all pathnames matching specified pattern
import matplotlib.pyplot as plt


def get_shortest(fls):
    data_size = np.inf
    for c in fls:
        loaded = np.load(c).shape[0]
        if data_size >= loaded:
            data_size = loaded
    return data_size


class DataGenerator(Sequence):
    """
    Class that instantiates Keras data generators.
    """

    def __init__(self, X_files, y_files, dims=(80, 80), n_channels=3):
        """
        Initialize the data generator instance.
        :param X_files: input file names
        :param y_files: output (label, target) file names
        :param batch_size: size of the data to be generated
        :param dims: image data dimensions (height x width)
        :param n_channels: image data channels (default 3 (RGB))
        """
        self.dims = dims
        self.X_files = X_files
        self.y_files = y_files
        self.n_batches = len(X_files)
        self.n_channels = n_channels

    def __len__(self):
        """
        Denote the number of batches per epoch.
        :return: number of batches per epoch (number of the batched input/output files)
        """
        return len(self.X_files)

    def get_filenames(self, X_dir, y_dir):
        """
        Populate the filename lists.
        :param X_dir: input files directory
        :param y_dir: output (label, target) files directory
        :return: total number of files (number of batches per epoch) in the parameter directories
        """
        self.X_files = glob(f'{X_dir}*.npy')
        self.y_files = glob(f'{y_dir}*.npy')

        return len(self.X_files)

    def __getitem__(self, index):
        """
        Generate the input and output data files at the given index and load.
        :param index: current data file index
        :return: input and output data files at the current index
        """
        X = np.load(self.X_files[index])
        X = X.transpose([0, 2, 3, 1])

        y = np.load(self.y_files[index])
        # for i in range(X.shape[0]):
        #     if y[i] < 0.48:
        #         plt.imshow(X[i])
        #         plt.show()
        #         print('\n')
        #         print(y[i])
        #         input()

        return X, y

    def on_epoch_end(self):
        pass

    def __data_generation(self):
        pass

