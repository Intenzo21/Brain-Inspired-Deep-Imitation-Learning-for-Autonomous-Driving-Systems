"""
Animate the HDF5-formatted comma.ai dataset recordings.

Adopted in splitting the dataset into sunny, cloudy and night time videos.
"""

import argparse
from glob import glob
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from tqdm import tqdm

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='HDF5 camera file animation program.')

# Fill the ArgumentParser object with information about program arguments.
parser.add_argument(
    'folder_path',
    type=str,
    help='camera frames folder path'
)

# Parse the arguments
args = parser.parse_args()

cam_fls = sorted(glob(args.folder_path))

fig = plt.figure()

# Loop over the camera HDF5 files, animate the middle 500 frames
# and save them as MP4 files
for c in cam_fls:
    name = c[-21:-3]
    cam_fl = h5py.File(c, 'r')['X']  # Open the camera file for reading
    mid = cam_fl.shape[0]//2
    frames = []
    for i, fr in tqdm(enumerate(cam_fl[mid-250:mid+250])):
        frames.append([plt.imshow(fr.transpose(1, 2, 0))])

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                    repeat_delay=1000)

    ani.save(f'recs/movie_{name}_new.mp4')
