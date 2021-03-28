import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

INPUT_SHAPE = (80, 120, 3)  # Image input shape


def scale_data(data, ds_min=None, ds_max=None):
    """
    Scale (normalize) the data.
    :param ds_min:
    :param ds_max:
    :param data: data list
    :return: normalized data list
    """
    # Check if minimum and maximum dataset value parameters are provided
    # If not take the current dataset file minimum and maximum values
    if all(i is None for i in [ds_min, ds_max]):
        ds_min = np.min(data)
        ds_max = np.max(data)
    data = (data - ds_min) / (ds_max - ds_min)
    return data


def standardize_data(data):
    data_mean = np.mean(data)
    data_std = np.std(data)
    data = (data - data_mean) / data_std
    return data


def save_data(file_name, **kwargs):
    """
    Save the data to a NumPy zip file.
    :param file_name:
    :param kwargs:
    :return:
    """
    np.savez(file_name, **kwargs)


def npy_chunks(data, outer_folder, inner_folder, chunk_size=32, chunk_start=0):
    """
    Divide data into chunks.
    :param data: data file
    :param folder_name: name of the folder where the chunks will be stored
    :param file_name: name of the chunk file
    :param chunk_size: size of each chunk of data
    :param chunk_start: used in enumerating the chunks (starting number for the file name)
    :return: number of the last chunk that has been saved
    """
    # chunk_start += 1
    file_name = f'{chunk_size}_{inner_folder}'
    path = f'{chunk_size}_batched_{outer_folder}/{inner_folder}'
    saved = 0
    for idx, i in tqdm(enumerate(range(0, data.shape[0], chunk_size))):
        chunk = data[i:i + chunk_size]
        # Discard the last chunk if it is less than the chunk (batch) size.
        # That is to keep the npy file sizes consistent.
        # Worst case a loss of (batch size * num of files) - num of files frames
        # Number of frames in the whole dataset: 522434
        # Number of files: 11
        # Batch size: 64
        # Frames lost (worst case)=(64 * 11) - 11=693
        # Percentage ~ 0.133% < 1%
        if chunk.shape[0] < chunk_size:
            continue
        try:
            np.save(f'{path}/{file_name}_{idx + chunk_start}', chunk)
        except FileNotFoundError:
            os.makedirs(f'{path}')
            np.save(f'{path}/{file_name}_{idx + chunk_start}', chunk)

        saved = idx
    return saved + 1  # To avoid same chunk file name issues (last and first chunks
    # from 2 consecutive .h5 recording files i.e. last chunk file name is 3060 so we increment by 1 in order to start
    # from 3061)


def plot_data(data, title, xlabel, ylabel):
    plt.figure(figsize=(10, 8))
    plt.plot(data, color='g')  # peak at 13850
    plt.title(title, weight='bold', fontsize=16)
    plt.xlabel(xlabel, weight='bold', fontsize=14)
    plt.ylabel(ylabel, weight='bold', fontsize=14)
    plt.xticks(weight='bold', fontsize=12, rotation=45)
    plt.yticks(weight='bold', fontsize=12)
    plt.grid(color='y', linewidth=0.5)
    plt.show()
