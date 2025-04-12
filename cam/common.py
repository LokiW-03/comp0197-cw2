# I acknowledge the use of ChatGPT (version GPT-4o, OpenAI, https://chatgpt.com/) for assistance in debugging and
# writing docstrings.

import os

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 30
LR = 5e-5
NUM_CLASSES = 37
MODEL_SAVE_PATH = f"{os.path.dirname(os.path.abspath(__file__))}/saved_models"
TMP_OUTPUT_PATH = "output"
NUM_SAMPLES = 16
WORKERS = 4


CAM_THRESHOLD = {
    "resnet": [0.25, 0.325],
    "resnet_drs": [0.1, 0.3]
}
