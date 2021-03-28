from glob import glob

import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from tqdm import tqdm

cam_fls = sorted(glob("./comma-dataset/cam/*.h5"))

fig = plt.figure()

for c in cam_fls:
    name = c[-21:-3]
    cam_fl = h5py.File(c, 'r')['X']  # Open the camera file for reading
    mid = cam_fl.shape[0]//2
    frames = []
    for i, fr in tqdm(enumerate(cam_fl[mid-250:mid+250])):
        frames.append([plt.imshow(fr.transpose(1, 2, 0))])

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                    repeat_delay=1000)

    ani.save(f'recs/movie_{name}.mp4')