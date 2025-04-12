# I acknowledge the use of ChatGPT (version GPT-4o, OpenAI, https://chatgpt.com/) for assistance in debugging and
# writing docstrings.

import random

random.seed(42)

def generate_refined_cam_threshold_space(base_low=0.25, base_high=0.325, space_size=10):
    """
    Generate a refined search space of (low, high) threshold pairs for CAM-based pseudo mask generation.

    Args:
        base_low (float, optional): Central value for the lower threshold in the fine search.
            Defaults to 0.25.
        base_high (float, optional): Central value for the upper threshold in the fine search.
            Defaults to 0.325.
        space_size (int, optional): Number of threshold pairs to sample and return. Defaults to 10.

    Returns:
        List[Tuple[float, float]]: A list of `(low_threshold, high_threshold)` pairs for use in
        pseudo-label generation with CAM. These pairs satisfy interval constraints and include
        both fine-grained and distribution-informed candidates.
    """

    # Core fine search area (around the current optimal solution)
    fine_low = [round(base_low + i*0.05, 3) for i in range(-2, 3)]
    fine_high = [round(base_high + i*0.05, 3) for i in range(-2, 3)]

    # Extended area based on distribution characteristics
    distribution_based_low = [0.18, 0.22, 0.28]  # Covers around the 25th percentile and mean offset
    distribution_based_high = [0.30, 0.35, 0.38]  # Covers the extended 75th percentile

    # Construct the final search space (remove duplicates)
    all_low = sorted(list(set(fine_low + distribution_based_low)))
    all_high = sorted(list(set(fine_high + distribution_based_high)))

    # Generate valid combinations
    valid_pairs = []
    for low in all_low:
        for high in all_high:
            # Maintain minimum interval constraints
            if 0.05 < (high - low) < 0.15:  # Based on interval analysis of successful cases
                valid_pairs.append((low, high))

    # Add special candidates (based on peak intervals in the distribution histogram)
    special_candidates = [
        (0.20, 0.30),  # Covers the high-density bins in the [0.2, 0.3) interval
        (0.25, 0.35),  # Extends the current optimal interval
        (0.22, 0.32)   # Offset test
    ]

    # Combine and remove duplicates
    all_candidates = sorted(list(set(valid_pairs + special_candidates)))

    # Randomly sample from the candidates
    random.shuffle(all_candidates)
    refined_candidates = all_candidates[:space_size]
    return refined_candidates
