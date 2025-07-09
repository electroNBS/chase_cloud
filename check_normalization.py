# check_normalization.py
import os
import rioxarray
import numpy as np
from tqdm import tqdm

# Configure this to match your setup
DATA_PATH = "data/raw_insat"
CHANNELS = ['MIR', 'WV', 'VIS', 'TIR1', 'TIR2', 'SWIR'] # List all channels you use

# Initialize dictionaries to hold the stats
channel_mins = {ch: float('inf') for ch in CHANNELS}
channel_maxs = {ch: float('-inf') for ch in CHANNELS}

print(f"Scanning all files in {DATA_PATH} to calculate true normalization stats...")
files_to_check = [f for f in os.listdir(DATA_PATH) if f.endswith('.tif')]

for filename in tqdm(files_to_check, desc="Analyzing Images"):
    try:
        # Extract channel from filename
        channel_name = filename.split('_')[-1].split('.')[0]
        if channel_name not in CHANNELS:
            continue
            
        fpath = os.path.join(DATA_PATH, filename)
        with rioxarray.open_rasterio(fpath) as rds:
            data = rds.values.squeeze()
            
            # Update min and max for this channel
            current_min = np.min(data)
            current_max = np.max(data)
            
            if current_min < channel_mins[channel_name]:
                channel_mins[channel_name] = current_min
            if current_max > channel_maxs[channel_name]:
                channel_maxs[channel_name] = current_max
    except Exception as e:
        print(f"\nCould not process {filename}. Error: {e}")

print("\n--- Analysis Complete ---")
print("Copy the following dictionary into your 'src/utils.py' file to replace CHANNEL_STATS:\n")

# Print the final dictionary in a copy-paste friendly format
print("CHANNEL_STATS = {")
for ch in CHANNELS:
    print(f"    '{ch}': {{'min': {channel_mins[ch]}, 'max': {channel_maxs[ch]}}},")
print("}")