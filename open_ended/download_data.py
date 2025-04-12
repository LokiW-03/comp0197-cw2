# I acknowledge the use of ChatGPT (version GPT-4o, OpenAI, https://chatgpt.com/) for assistance in debugging and
# writing docstrings.

# download_data.py

import os
import logging
from data_utils.data import download_oxford_pet_oeq

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Root directory where data *should end up*
FINAL_DATA_ROOT = "./data"
# Directory for weak labels (will be created, but file is generated elsewhere)
# --- End Configuration ---


def main():
    """
    Main function to orchestrate the data download, restructuring, and setup.
    """
    # Use FINAL_DATA_ROOT as the place torchvision downloads *into*.
    # The restructuring step will then fix the layout *within* this directory.
    if os.path.exists(FINAL_DATA_ROOT):
        logging.info("Data already exist")
    else:
        if download_oxford_pet_oeq(FINAL_DATA_ROOT):
            logging.info("Download step successful.")
        else:
            logging.error("Dataset download failed. Cannot proceed with restructuring.")


if __name__ == "__main__":
    main()