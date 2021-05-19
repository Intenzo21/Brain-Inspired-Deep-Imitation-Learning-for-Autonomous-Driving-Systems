"""
Project constants module.

All the constants are included in this script.
"""

# ======================================= Comma.ai dataset test data size and random state constants
TEST_SIZE = .2
RANDOM_STATE = 42

# ======================================= Comma.ai dataset training data and label directories
TRAIN_DATA_DIR = "./dataset/64_batched_sunny/frames/*.npy"
TRAIN_LABELS_FL_DIR = "./dataset/64_batched_sunny/angles/*.npy"

# ======================================= Cloudy weather evaluation data and label directories
C_TEST_DATA_DIR = "./dataset/64_batched_cloudy/frames/*.npy"
C_TEST_LABELS_FL_DIR = "./dataset/64_batched_cloudy/angles/*.npy"

# ======================================= Night time evaluation data and label directories
N_TEST_DATA_DIR = "./dataset/64_batched_night/frames/*.npy"
N_TEST_LABELS_FL_DIR = "./dataset/64_batched_night/angles/*.npy"

# ======================================= Comma.ai dataset model training hyperparameters
B_SIZE = 64
NB_EPOCHS = 10
LR = 5e-06  # Since trains the model to the best val_loss in 10 epochs (model starts overfitting)
VERBOSITY = 1


INPUT_SHAPE = (80, 120, 3)  # Image input shape
RESIZE_DIMS = [80, 120]
CROP_SIZE = [40, -40]