#!/usr/bin/env python3
"""
Script to parse segmentation training log files and extract key information.

The script extracts:
  • A trial header (e.g. "Trial 0:" or "Trial 1:")
  • The hyperparameters (from the "Parameters:" line)
  • The number of training batches (if available)
  • Epoch‐by‐epoch summaries (the "Epoch X/10" blocks, with the separate training and test metrics)
  • The best model information (the epoch at which the best performance was achieved and its corresponding metrics)

Usage:
    python3 extract_log_info.py effunet.log segnet.log segnext.log unet.log
"""

import sys
import re
import ast
import json

def safe_eval(dict_str):
    """Safely evaluate a dictionary string using ast.literal_eval."""
    try:
        return ast.literal_eval(dict_str)
    except Exception as e:
        return {"error": str(e)}

def parse_log_file(filepath):
    """Parse a log file and extract the important information."""
    data = {"filename": filepath, "trials": []}
    current_trial = None

    with open(filepath, "r") as f:
        lines = f.readlines()

    # Patterns to detect various parts of the log
    trial_pattern       = re.compile(r"Trial\s+(\d+):")
    param_pattern       = re.compile(r"Parameters:\s*(\{.*\})")
    batch_pattern       = re.compile(r"Number of train batches:\s*(\d+)")
    epoch_pattern       = re.compile(r"Epoch\s+(\d+)/\d+")
    train_pattern       = re.compile(r"Train\s*->\s*(\{.*\})")
    test_pattern        = re.compile(r"Test\s*->\s*(\{.*\})")
    best_model_pattern  = re.compile(r"Best model found at epoch\s+(\d+).*IoU\s+([\d\.]+)")

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Start of a new trial
        trial_match = trial_pattern.search(line)
        if trial_match:
            current_trial = {
                "trial_number": int(trial_match.group(1)),
                "parameters": {},
                "num_train_batches": None,
                "epochs": [],
                "best_model": {}
            }
            data["trials"].append(current_trial)
            i += 1
            continue

        if current_trial is not None:
            # Extract hyperparameters
            param_match = param_pattern.search(line)
            if param_match:
                param_str = param_match.group(1)
                current_trial["parameters"] = safe_eval(param_str)
                i += 1
                continue

            # Extract number of training batches
            batch_match = batch_pattern.search(line)
            if batch_match:
                current_trial["num_train_batches"] = int(batch_match.group(1))
                i += 1
                continue

            # Extract epoch information
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                epoch_num = int(epoch_match.group(1))
                epoch_entry = {"epoch": epoch_num, "train": None, "test": None}
                j = i + 1
                # Loop to capture the train and test info in this epoch block
                while j < len(lines):
                    subline = lines[j].strip()
                    t_match = train_pattern.search(subline)
                    if t_match:
                        train_dict_str = t_match.group(1)
                        epoch_entry["train"] = safe_eval(train_dict_str)
                    te_match = test_pattern.search(subline)
                    if te_match:
                        test_dict_str = te_match.group(1)
                        epoch_entry["test"] = safe_eval(test_dict_str)
                        # Once both are found or a break line is encountered, exit the loop for this epoch
                        break
                    if "--------------------------------------------------" in subline:
                        break
                    j += 1
                current_trial["epochs"].append(epoch_entry)
                i = j + 1
                continue

            # Extract best model information
            best_match = best_model_pattern.search(line)
            if best_match:
                best_epoch = int(best_match.group(1))
                best_iou = float(best_match.group(2))
                best_details = {"epoch": best_epoch, "best_iou": best_iou, "test": None, "train": None}
                # Look ahead for the test and train metric dictionaries
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    t_match = test_pattern.search(next_line)
                    if t_match:
                        best_details["test"] = safe_eval(t_match.group(1))
                if i + 2 < len(lines):
                    next_line = lines[i + 2].strip()
                    tr_match = train_pattern.search(next_line)
                    if tr_match:
                        best_details["train"] = safe_eval(tr_match.group(1))
                current_trial["best_model"] = best_details
        i += 1

    return data

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 extract_log_info.py <logfile1> [<logfile2> ...]")
        sys.exit(1)

    all_data = []
    for filepath in sys.argv[1:]:
        parsed = parse_log_file(filepath)
        all_data.append(parsed)

    print(json.dumps(all_data, indent=2))

if __name__ == "__main__":
    main()
