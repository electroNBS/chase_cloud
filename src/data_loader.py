# src/data_loader.py

import torch
from torch.utils.data import Dataset
import os
import numpy as np
from datetime import datetime, timedelta
import rioxarray  # Use rioxarray for GeoTIFF files

# import cv2
from tqdm import tqdm

# Import the normalization function from our utils file
from .utils import normalize, resize_and_pad


class INSATCloudDataset(Dataset):
    def __init__(
        self,
        root_dir,
        channels,
        input_frames,
        future_frames,
        image_size,
        temporal_res_mins=30,
    ):
        self.root_dir = root_dir
        self.channels = channels
        self.input_frames = input_frames
        self.future_frames = future_frames
        self.image_size = image_size
        self.temporal_res = timedelta(minutes=temporal_res_mins)
        self.sequence_length = self.input_frames + self.future_frames

        print("Finding valid temporal sequences (checking file existence)...")
        self.sequences = self._create_sequences()
        if not self.sequences:
            raise ValueError(
                f"No valid, complete temporal sequences found in {self.root_dir}. Check data and config."
            )
        print(f"Found {len(self.sequences)} complete and valid sequences.")

    # --- MODIFIED SECTION ---
    def _create_sequences(self):
        all_files = os.listdir(self.root_dir)
        timestamps = set()
        for f in all_files:
            if not f.endswith(".tif"):
                continue
            try:
                parts = f.split("_")
                dt = datetime.strptime(f"{parts[1]}{parts[2]}", "%d%b%Y%H%M")
                timestamps.add(dt)
            except (IndexError, ValueError):
                continue

        sorted_timestamps = sorted(list(timestamps))
        valid_sequences = []

        # tqdm provides a progress bar, which is helpful here
        for i in tqdm(
            range(len(sorted_timestamps) - self.sequence_length + 1),
            desc="Verifying sequences...",
        ):
            sequence_dts = sorted_timestamps[i : i + self.sequence_length]

            # 1. Check for temporal continuity
            is_continuous = all(
                sequence_dts[j + 1] - sequence_dts[j] == self.temporal_res
                for j in range(len(sequence_dts) - 1)
            )
            if not is_continuous:
                continue

            # 2. Check if all files for this sequence actually exist on disk
            all_files_exist = True
            for dt in sequence_dts:
                for channel in self.channels:
                    # Construct filename
                    filename = f"3DIMG_{dt.strftime('%d%b%Y').upper()}_{dt.strftime('%H%M')}_L1C_SGP_V01R00_IMG_{channel}.tif"
                    # Use os.path.join to create a platform-independent path
                    fpath = os.path.join(self.root_dir, filename)
                    if not os.path.exists(fpath):
                        all_files_exist = False
                        break  # No need to check other channels
                if not all_files_exist:
                    break  # No need to check other datetimes in this sequence

            # Only if all checks pass, add the sequence to our list
            if all_files_exist:
                valid_sequences.append(sequence_dts)

        return valid_sequences

    # --- END MODIFIED SECTION ---

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_dts = self.sequences[idx]

        # This function processes a list of datetimes into a tensor
        def _load_frames(datetimes):
            frames_list = []
            for dt in datetimes:
                frame_channels = []
                for channel in self.channels:
                    filename = f"3DIMG_{dt.strftime('%d%b%Y').upper()}_{dt.strftime('%H%M')}_L1C_SGP_V01R00_IMG_{channel}.tif"
                    fpath = os.path.join(self.root_dir, filename)
                    try:
                        # *** MODIFIED SECTION: Use rioxarray to open GeoTIFF ***
                        with rioxarray.open_rasterio(fpath) as rds:
                            # .values returns a NumPy array. Squeeze removes single-dimensions.
                            # Geotiffs might be (1, H, W), so we get (H, W).
                            data = rds.values.squeeze()

                        processed_data = resize_and_pad(data, self.image_size)
                        normalized_data = normalize(processed_data, channel)
                        # Resize to the desired model input size
                        # data = cv2.resize(
                        #     data,
                        #     (self.image_size, self.image_size),
                        #     interpolation=cv2.INTER_AREA,
                        # )
                        frame_channels.append(normalized_data)

                    except Exception as e:
                        print(
                            f"Warning: Could not load or process {fpath}. Error: {e}. Replacing with zeros."
                        )
                        frame_channels.append(
                            np.zeros(
                                (self.image_size, self.image_size), dtype=np.float32
                            )
                        )

                # Stack channels for a single time step: (num_channels, H, W)
                frames_list.append(np.stack(frame_channels, axis=0))
            return torch.from_numpy(np.array(frames_list)).float()

        past_frames_tensor = _load_frames(sequence_dts[: self.input_frames])
        future_frames_tensor = _load_frames(sequence_dts[self.input_frames :])

        # Concatenate past frames along the channel dimension for conditioning
        # Shape becomes: (input_frames * num_channels, H, W)
        condition_tensor = past_frames_tensor.flatten(start_dim=0, end_dim=1)

        # Target is the first future frame. Shape: (num_channels, H, W)
        target_tensor = future_frames_tensor[0]

        return condition_tensor, target_tensor
