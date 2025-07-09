# src/utils.py
import numpy as np
import cv2

# --- Helper function for normalization ---
# *** IMPORTANT: Update the channel names and stats for your data ***
# MIR is Mid-Infrared, which corresponds to TIR.
# Channel statistics for normalization.
# IMPORTANT: You should calculate the true min/max from your dataset for best results.
CHANNEL_STATS = {
    "MIR": {"min": 0, "max": 1023},
    "WV": {"min": 745, "max": 1023},
    "VIS": {"min": 0, "max": 1023},
    "TIR1": {"min": 236, "max": 1023},
    "TIR2": {"min": 271, "max": 1023},
    "SWIR": {"min": 0, "max": 1023},
}


def normalize(data, channel):
    """Normalizes data to [-1, 1] range."""
    stats = CHANNEL_STATS.get(channel)
    if not stats:
        min_val, max_val = data.min(), data.max()
    else:
        min_val, max_val = stats["min"], stats["max"]

    if max_val == min_val:
        return np.zeros_like(data, dtype=np.float32)

    data = (data - min_val) / (max_val - min_val)
    return (2 * data - 1).astype(np.float32)


# --- THIS IS THE CORRECTED FUNCTION ---
def denormalize(data):
    """Denormalizes data from [-1, 1] back to [0, 1] and clips it to ensure validity."""
    data = (data + 1) / 2  # Convert from [-1, 1] to [0, 1]
    # Clip the values to handle any floating point inaccuracies.
    return np.clip(data, 0.0, 1.0)


# --- END CORRECTION ---


def resize_and_pad(img, target_size):
    """
    Resizes an image to a target size while maintaining aspect ratio by padding.
    """
    old_h, old_w = img.shape
    scale = target_size / max(old_h, old_w)

    # Calculate new dimensions
    new_w, new_h = int(old_w * scale), int(old_h * scale)

    # Resize the image
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a new image with a black background (zero-padded)
    padded_img = np.zeros((target_size, target_size), dtype=resized_img.dtype)

    # Calculate position to paste the resized image
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2

    # Paste the resized image onto the center of the padded image
    padded_img[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized_img

    return padded_img
