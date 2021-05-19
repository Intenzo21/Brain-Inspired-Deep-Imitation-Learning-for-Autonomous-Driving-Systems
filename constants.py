"""
Project constants module.

All the constants are included in this script.
"""

# ======================================= drive_model.py
SPEED = 9  # Default desired car speed
KP = .1  # Proportional gain constant
KI = .002  # integral gain constant

# ======================================= Udacity dataset input image dimensions and shape
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

# ======================================= Udacity dataset training data and label directories
TRAIN_DATA_DIR = './data/lakeside_map/'
TRAIN_LABELS_FL_DIR = './data/lakeside_map/driving_log.csv'

# ======================================= Udacity dataset evaluation data and label directories
TEST_DATA_DIR = './data/jungle_map/'
TEST_LABELS_FL_DIR = './data/jungle_map/driving_log.csv'

# ======================================= Udacity dataset training batch size, number of epochs and learning rate
B_SIZE = 40
NB_EPOCHS = 10
LR = 1.0e-4
