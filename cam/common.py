import os

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 20
LR = 1e-3
NUM_CLASSES = 37
MODEL_SAVE_PATH = f"{os.path.dirname(os.path.abspath(__file__))}/saved_models"
TMP_OUTPUT_PATH = "output"
NUM_SAMPLES = 16
WORKERS = 4