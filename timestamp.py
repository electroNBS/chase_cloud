# find_good_timestamp.py

import os
from datetime import datetime, timedelta

# --- CONFIGURE THIS TO MATCH YOUR src/config.py ---
DATA_PATH = "data/raw_insat"
CHANNELS = ['MIR', 'WV', 'VIS', 'TIR1', 'TIR2', 'SWIR'] # Use all the channels you are training on
INPUT_FRAMES = 4 # Use the same number of frames as your model
# --------------------------------------------------

# Get a list of all unique timestamps from your filenames
timestamps = set()
for f in os.listdir(DATA_PATH):
    if not f.endswith('.tif'): continue
    try:
        parts = f.split('_')
        dt = datetime.strptime(f"{parts[1]}{parts[2]}", "%d%b%Y%H%M")
        timestamps.add(dt)
    except (IndexError, ValueError): continue

# Search backwards from the most recent timestamp
print("Searching for a valid sequence with all required files...")
for end_dt in sorted(list(timestamps), reverse=True):
    sequence_dts = [end_dt - timedelta(minutes=30 * i) for i in range(INPUT_FRAMES)]
    # Also check for the ground truth file
    all_dts_to_check = sequence_dts + [end_dt + timedelta(minutes=30)]
    
    all_files_exist = True
    for dt in all_dts_to_check:
        for channel in CHANNELS:
            filename = f"3DIMG_{dt.strftime('%d%b%Y').upper()}_{dt.strftime('%H%M')}_L1C_SGP_V01R00_IMG_{channel}.tif"
            fpath = os.path.join(DATA_PATH, filename)
            if not os.path.exists(fpath):
                all_files_exist = False
                break
        if not all_files_exist:
            break
            
    if all_files_exist:
        print("\n--- SUCCESS! ---")
        print(f"Found a fully complete sequence ending at: {end_dt}")
        print("Use this as the default date and time in your Streamlit app.")
        print("----------------\n")
        break