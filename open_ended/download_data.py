# download_data.py

import os
import logging
import shutil  # For moving files/directories
import tarfile # For checking archive integrity (optional but good practice)
from torchvision.datasets import OxfordIIITPet
from torchvision.datasets.utils import download_url, check_integrity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Root directory where data *should end up*
FINAL_DATA_ROOT = "./data"
# Subdirectory name torchvision creates by default
TORCHVISION_SUBDIR = "oxford-iiit-pet"
# Directory for weak labels (will be created, but file is generated elsewhere)
# Optionally remove downloaded tar.gz files after extraction and move
CLEANUP_ARCHIVES = True
# --- End Configuration ---

# TODO: edit final_data_root and remove resturcturing
def restructure_directory(download_root, final_root, torchvision_subdir):
    """
    Moves contents from the torchvision subdirectory to the final root
    and cleans up.
    """
    intermediate_dir = os.path.join(download_root, torchvision_subdir)
    final_images_dir = os.path.join(final_root, 'images')
    final_annotations_dir = os.path.join(final_root, 'annotations')

    if not os.path.isdir(intermediate_dir):
        logging.warning(f"Intermediate directory '{intermediate_dir}' not found. Assuming data is already structured or download failed.")
        # Check if final structure already exists, maybe from a previous run
        if os.path.isdir(final_images_dir) and os.path.isdir(final_annotations_dir):
             logging.info(f"Final directories '{final_images_dir}' and '{final_annotations_dir}' already exist. Skipping restructure.")
             return True # Indicate success as desired structure exists
        else:
             logging.error(f"Neither intermediate nor final directories found. Download likely failed.")
             return False # Indicate failure

    # Check if the final structure already exists before moving
    if os.path.isdir(final_images_dir) or os.path.isdir(final_annotations_dir):
         logging.info(f"It seems '{final_images_dir}' or '{final_annotations_dir}' already exists. Skipping move operation.")
         # Optionally, still perform cleanup of the intermediate dir if it exists
         try:
             if os.path.isdir(intermediate_dir):
                 logging.info(f"Removing intermediate directory structure: '{intermediate_dir}'")
                 shutil.rmtree(intermediate_dir) # Use rmtree as it might contain archives
             # Also check for archives directly in download_root if cleanup is desired
             if CLEANUP_ARCHIVES:
                 img_tar = os.path.join(download_root, 'images.tar.gz')
                 ann_tar = os.path.join(download_root, 'annotations.tar.gz')
                 if os.path.isfile(img_tar): os.remove(img_tar)
                 if os.path.isfile(ann_tar): os.remove(ann_tar)

         except OSError as e:
            logging.warning(f"Could not completely clean up intermediate directory/files: {e}")
         return True # Assume prior success if final dirs exist

    logging.info(f"Moving contents from '{intermediate_dir}' to '{final_root}'...")

    try:
        # Move contents
        for item_name in ['images', 'annotations']:
            src_path = os.path.join(intermediate_dir, item_name)
            dest_path = os.path.join(final_root, item_name)
            if os.path.exists(src_path):
                logging.info(f"Moving '{src_path}' to '{dest_path}'")
                shutil.move(src_path, dest_path)
            else:
                logging.warning(f"Source path '{src_path}' not found during move.")

        # Cleanup
        logging.info(f"Cleaning up intermediate directory '{intermediate_dir}'...")
        # Remove original archives if desired (they might be in intermediate_dir or download_root)
        if CLEANUP_ARCHIVES:
            img_tar_inter = os.path.join(intermediate_dir, 'images.tar.gz')
            ann_tar_inter = os.path.join(intermediate_dir, 'annotations.tar.gz')
            img_tar_root = os.path.join(download_root, 'images.tar.gz')
            ann_tar_root = os.path.join(download_root, 'annotations.tar.gz')

            for f in [img_tar_inter, ann_tar_inter, img_tar_root, ann_tar_root]:
                 if os.path.isfile(f):
                     try:
                         logging.info(f"Removing archive file: '{f}'")
                         os.remove(f)
                     except OSError as e:
                         logging.warning(f"Could not remove archive '{f}': {e}")

        # Remove the now potentially empty intermediate directory
        try:
            # Check again if it exists before removing
            if os.path.isdir(intermediate_dir):
                 # Ensure it's empty before rmdir, or use rmtree if archives might remain
                 if not os.listdir(intermediate_dir):
                      os.rmdir(intermediate_dir)
                      logging.info(f"Removed empty intermediate directory: '{intermediate_dir}'")
                 else:
                      logging.warning(f"Intermediate directory '{intermediate_dir}' not empty after move, attempting rmtree.")
                      shutil.rmtree(intermediate_dir)


        except OSError as e:
            logging.warning(f"Could not remove intermediate directory '{intermediate_dir}': {e}")

        logging.info("Restructure and cleanup complete.")
        return True

    except Exception as e:
        logging.error(f"An error occurred during directory restructuring: {e}", exc_info=True)
        return False


# TODO: check model/data.py and see if we can combine the two together
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
        logging.info(f"Initial download target directory: {os.path.abspath(download_root)}")
        logging.info("This may take a while depending on your internet connection.")

        _ = OxfordIIITPet(root=download_root, split="trainval", target_types="segmentation", download=True)

        logging.info("Dataset download/verification step complete.")
        # Further verification happens implicitly during the restructuring phase
        return True

    except Exception as e:
        logging.error(f"An error occurred during dataset download: {e}", exc_info=True)
        logging.error("Please check your internet connection, disk space, and permissions.")
        return False

def setup_directories(final_data_dir, weak_label_dir):
    """
    Creates the necessary base directories if they don't exist.
    """
    logging.info("Setting up necessary directory structure...")
    try:
        # Ensure the *final* data directory exists for restructuring into
        os.makedirs(final_data_dir, exist_ok=True)
        logging.info(f"Ensured final data directory exists: '{os.path.abspath(final_data_dir)}'")
        os.makedirs(weak_label_dir, exist_ok=True)
        logging.info(f"Ensured weak label directory exists: '{os.path.abspath(weak_label_dir)}'")
    except OSError as e:
        logging.error(f"Error creating directories: {e}")
        return False
    return True

def main():
    """
    Main function to orchestrate the data download, restructuring, and setup.
    """
    # Use FINAL_DATA_ROOT as the place torchvision downloads *into* initially.
    # The restructuring step will then fix the layout *within* this directory.
    download_target_root = FINAL_DATA_ROOT


    if download_oxford_pet(download_target_root):
        logging.info("Download step successful. Proceeding to restructure directories...")
        if restructure_directory(download_target_root, FINAL_DATA_ROOT, TORCHVISION_SUBDIR):
            logging.info("-" * 60)
            logging.info("Oxford-IIIT Pet dataset setup process finished successfully.")
            logging.info(f"Data should now be structured directly in: '{os.path.abspath(FINAL_DATA_ROOT)}'")
            logging.info("Expected final subdirectories: 'images/' and 'annotations/' (containing 'trimaps/')")
            logging.info("-" * 60)
        else:
            logging.error("Directory restructuring failed. Please check the logs.")
    else:
        logging.error("Dataset download failed. Cannot proceed with restructuring.")


if __name__ == "__main__":
    main()