from glob import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
from utils import plot_data

log_fls = glob("64_batched_night/angles/*.npy")
print(log_fls)
all_logs = []
for i, l in enumerate(log_fls):
    log = np.load(l)  # Open the log file for reading
    for i in range(log.shape[0]):
        all_logs.append(log[i])

print(np.max(all_logs), np.min(all_logs))
plot_data(all_logs, 'Steering angles', 'Frame', 'Steering angle (scaled)')
input()
