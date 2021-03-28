# Importing Python libraries
seed_value = 1337

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random

random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np

np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf

tf.random.set_seed(seed_value)  # tensorflow 2.x

import matplotlib.pyplot as plt
from glob import glob
from data_generator import DataGenerator
from kerasncp.tf import LTCCell
from keras.models import load_model
from sklearn.utils import shuffle

gpus = tf.config.experimental.list_physical_devices('GPU')
# To turn on memory growth for a specific GPU, we use the following code prior to allocating
# any tensors or executing any ops.
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Fetch the batched dataset frame and angle NumPy filenames
# Here we adopt the cloudy dataset recording files to evaluate
# the checkpoint models (best loss).
cam_files = sorted(glob("./dataset/64_batched_cloudy/frames/*.npy"))
log_files = sorted(glob("./dataset/64_batched_cloudy/angles/*.npy"))

# Shuffle the frame and angle filenames
cam_files, log_files = shuffle(cam_files, log_files, random_state=42)

# Initialize the test data generator
test_gen = DataGenerator(cam_files, log_files)

cam_files_night = sorted(glob("./dataset/64_batched_night/frames/*.npy"))
log_files_night = sorted(glob("./dataset/64_batched_night/angles/*.npy"))

cam_files_night, log_files_night = shuffle(cam_files_night, log_files_night, random_state=42)

test_gen_night = DataGenerator(cam_files_night, log_files_night)


def eval_model(model, model_name, gen):
    """
    Evaluate model and print the test loss
    :param tf.keras.Model model:
    :param string model_name: name of the model
    :param DataGenerator gen: DataGenerator instance (test data generator in this case)
    :return: None
    """
    print(f'\n{model_name} model evaluation:')
    res = model.evaluate(gen)
    print(f'Test loss (MSE): {res}')


def plot_angles(model_name, actual, predicted):
    """
    Plot the actual vs predicted steering angle values
    :param actual:
    :param predicted:
    :return:
    """
    plt.figure(figsize=(11, 6))
    plt.plot(actual, linestyle='solid', color='r')
    plt.plot(predicted, linestyle='dashed', color='b')
    plt.legend(['Actual', 'Predicted'], loc='best', prop={'size': 14})
    plt.title(f'{model_name} Actual vs Predicted Steering Angle Values', weight='bold', fontsize=16)
    plt.ylabel('Angle (scaled)', weight='bold', fontsize=14)
    plt.xlabel('Batch Sample no.', weight='bold', fontsize=14)
    plt.xticks(weight='bold', fontsize=12, rotation=45)
    plt.yticks(weight='bold', fontsize=12)
    plt.grid(color='y', linewidth='0.5')
    plt.show()


# Load and evaluate the CNN checkpoint model
CNN_path = 'checkpoints/sunny_dataset_same/CNN_5e-06/seeded_cpu/CNN_model-0.002091.h5'
CNN_model = load_model(CNN_path)
CNN_model.summary()

eval_model(CNN_model, 'CNN (Nvidia)', test_gen)  # 0.0022485055960714817 | Time: 25

# Predict steering angle output values using a single batch from the test data generator
prediction = CNN_model(test_gen[0][0])

y_actual = test_gen[0][1]  # Actual (expected) steering angle output values
y_pred = prediction  # Predicted steering angle output values

plot_angles('CNN', y_actual, y_pred)

# Load and evaluate the stacked CNN+NCP checkpoint model
stacked_path = 'checkpoints/sunny_dataset_same/stacked_5e-06/seeded_cpu/stacked_model-0.002154.h5'
stacked_model = load_model(stacked_path, custom_objects={'CustomLayer': LTCCell})
stacked_model.summary()

eval_model(stacked_model, 'Stacked (CNN+NCP)', test_gen)  # 0.002199082635343075 | Time: 31

# Predict steering angle output values using a single batch from the test data generator
prediction = stacked_model(test_gen[0][0])

y_actual = test_gen[0][1]  # Actual (expected) steering angle output values
y_pred = prediction  # Predicted steering angle output values

plot_angles('Stacked (CNN+NCP)', y_actual, y_pred)

# Load and evaluate the stacked CNN+NCP checkpoint model
stacked_dual_v2_path = 'checkpoints/sunny_dataset_same/stacked_dual_v2_5e-06/seeded_cpu/' \
                       'stacked_dual_v2_model-0.002150.h5'
stacked_dual_model_v2 = load_model(stacked_dual_v2_path, custom_objects={'CustomLayer': LTCCell})
stacked_dual_model_v2.summary()

eval_model(stacked_dual_model_v2, 'Stacked (CNN+dual NCP) v2', test_gen)  # 0.002199082635343075 | Time: 31

# Predict steering angle output values using a single batch from the test data generator
prediction = stacked_dual_model_v2(test_gen[0][0])

y_actual = test_gen[0][1]  # Actual (expected) steering angle output values
y_pred = prediction  # Predicted steering angle output values

plot_angles('Stacked (CNN+dual NCP) v2', y_actual, y_pred)

