#!/usr/bin/env python3
import json
import argparse

def load_trials(file_path):
    """Load trials data from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data[0]['trials']

def filter_trials(trials, trial_numbers):
    """
    Filter trials based on the trial numbers.
    Args:
        trials (list): List of trial dictionaries.
        trial_numbers (list): List of trial numbers to include.
    Returns:
        list: Filtered trial dictionaries.
    """
    return [trial for trial in trials if trial.get('trial_number') in trial_numbers]

def main():
    parser = argparse.ArgumentParser(
        description="Filter trials from a JSON file based on trial numbers."
    )
    parser.add_argument(
        '--file',
        type=str,
        default='parsed_segnet.txt',
        help='Path to the JSON file containing the trials data (default: parsed_segnet.txt)'
    )
    parser.add_argument(
        '--trials',
        type=int,
        nargs='+',
        required=True,
        help="List of trial numbers to extract. Example: --trials 0 2 5"
    )
    args = parser.parse_args()

    try:
        trials = load_trials(args.file)
    except Exception as e:
        print(f"Error loading file {args.file}: {e}")
        return

    filtered_trials = filter_trials(trials, args.trials)

    if filtered_trials:
        print(json.dumps(filtered_trials, indent=2))
    else:
        print("No trials matched the specified trial numbers.")

if __name__ == "__main__":
    main()
