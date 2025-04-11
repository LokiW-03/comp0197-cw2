# download_data.py

import os
import logging
import shutil  # For moving files/directories
from torchvision.datasets import OxfordIIITPet

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Root directory where data *should end up*
FINAL_DATA_ROOT = "./data"
# Directory for weak labels (will be created, but file is generated elsewhere)
# --- End Configuration ---



def download_oxford_pet(download_root):
    """
    Downloads the Oxford-IIIT Pet dataset using torchvision.
    Uses download_root as the target for the initial download.

    Args:
        download_root (str): The path where torchvision will initially place files.

    Returns:
        bool: True if download/verification was successful or data already exists,
              False otherwise.
    """
    logging.info(f"Checking for Oxford-IIIT Pet dataset in '{os.path.abspath(download_root)}'...")

    try:
        # We instantiate the dataset class primarily to trigger the download=True logic.
        # The actual dataset object isn't used further in this download script.
        # We use download_root here, which will contain the 'oxford-iiit-pet' subdir.
        logging.info("Attempting to download/verify Oxford-IIIT Pet dataset (images and annotations)...")
        logging.info(f"Download target directory: {os.path.abspath(download_root)}")
        logging.info("This may take a while depending on your internet connection.")

        _ = OxfordIIITPet(root=download_root, split="trainval", target_types="segmentation", download=True)

        logging.info("Dataset download/verification step complete.")
        # Further verification happens implicitly during the restructuring phase
        return True

    except Exception as e:
        logging.error(f"An error occurred during dataset download: {e}", exc_info=True)
        logging.error("Please check your internet connection, disk space, and permissions.")
        return False


def main():
    """
    Main function to orchestrate the data download, restructuring, and setup.
    """
    # Use FINAL_DATA_ROOT as the place torchvision downloads *into*.
    # The restructuring step will then fix the layout *within* this directory.
    download_target_root = FINAL_DATA_ROOT

    if download_oxford_pet(download_target_root):
        logging.info("Download step successful.")
    else:
        logging.error("Dataset download failed. Cannot proceed with restructuring.")


if __name__ == "__main__":
    main()