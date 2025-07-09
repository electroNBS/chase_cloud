import streamlit as st
import torch
import numpy as np
import os
from datetime import datetime, time, timedelta
import sys
import rioxarray

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
ABS_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw_insat")

try:
    from src.train import CloudChaseModule
    from src.config import CHANNELS, INPUT_FRAMES, IMAGE_SIZE
    from src.utils import normalize, resize_and_pad, denormalize
    import rioxarray
    MODEL_CODE_AVAILABLE = True
    
except ImportError as e:
    MODEL_CODE_AVAILABLE = False
    IMPORT_ERROR_MESSAGE = f"Failed to import necessary libraries. Please run 'pip install -r requirements.txt'. Error: {e}"

st.set_page_config(page_title="Chase the Cloud", page_icon="☁️", layout="wide")

CHECKPOINT_FILE = os.path.join(
    PROJECT_ROOT, "checkpoints", "cloud-chase-best-epoch=04-train_loss=0.82.ckpt"
)  # <-- UPDATE THIS EVERY TIME YOU TRAIN THE MODEL WITH THE LATEST CHECKPOINT


@st.cache_resource
def load_model(checkpoint_path):
    if not (os.path.exists(checkpoint_path) and MODEL_CODE_AVAILABLE):
        return None
    if "dummy" in os.path.basename(checkpoint_path).lower():
        return "dummy"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CloudChaseModule.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    return model


# Data Loading Function with a "Fill the Gaps" Logic
def load_data_with_gaps_filled(end_dt):
    sequence_dts = [
        end_dt - timedelta(minutes=30 * i) for i in range(INPUT_FRAMES - 1, -1, -1)
    ]
    ground_truth_dt = end_dt + timedelta(minutes=30)

    st.info(
        f"Attempting to load data for sequence ending at {end_dt.strftime('%H:%M')} UTC."
    )

    condition_frames = []
    for dt in sequence_dts:
        frame_channels = []
        for channel in CHANNELS:
            filename = f"3DIMG_{dt.strftime('%d%b%Y').upper()}_{dt.strftime('%H%M')}_L1C_SGP_V01R00_IMG_{channel}.tif"
            fpath = os.path.join(ABS_DATA_PATH, filename)

            if os.path.exists(fpath):
                with rioxarray.open_rasterio(fpath) as rds:
                    data = rds.values.squeeze()
                processed_data = resize_and_pad(data, IMAGE_SIZE)
            else:
                # This is a workaround, since our data is missing images, we insert a blank img
                st.warning(f"Missing file: {filename}. Using a blank image.")
                processed_data = np.zeros((IMAGE_SIZE, IMAGE_SIZE))

            normalized_data = normalize(processed_data, channel)
            frame_channels.append(normalized_data)
        condition_frames.append(np.stack(frame_channels, axis=0))

    condition_tensor = (
        torch.from_numpy(np.array(condition_frames)).float().flatten(0, 1).unsqueeze(0)
    )
    last_input_image = denormalize(condition_frames[-1])

    # Load ground truth (also with gap filling)
    gt_channels = []
    for channel in CHANNELS:
        filename = f"3DIMG_{ground_truth_dt.strftime('%d%b%Y').upper()}_{ground_truth_dt.strftime('%H%M')}_L1C_SGP_V01R00_IMG_{channel}.tif"
        fpath = os.path.join(ABS_DATA_PATH, filename)
        if os.path.exists(fpath):
            with rioxarray.open_rasterio(fpath) as rds:
                data = rds.values.squeeze()
            processed_data = resize_and_pad(data, IMAGE_SIZE)
        else:
            st.warning(f"Missing ground truth file: {filename}. Using a blank image.")
            processed_data = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
        gt_channels.append(denormalize(normalize(processed_data, channel)))
    ground_truth_image = np.stack(gt_channels, axis=0)

    return condition_tensor, last_input_image, ground_truth_image


# --- Main Application UI ---
st.title("☁️ Chase the Cloud: INSAT-3DR Cloud Motion Prediction")

st.write("--- DEBUG INFO ---")
st.write(f"Attempting to load checkpoint from this exact path: `{CHECKPOINT_FILE}`")
st.write(
    f"Does the file exist according to the OS? **{os.path.exists(CHECKPOINT_FILE)}**"
)
st.write("--- END DEBUG INFO ---")

if not MODEL_CODE_AVAILABLE:
    st.error(IMPORT_ERROR_MESSAGE)

model = load_model(CHECKPOINT_FILE)

if model is None:
    st.error(
        f"**Model checkpoint not found!** Please ensure a model exists at `{CHECKPOINT_FILE}`."
    )
else:
    if model != "dummy":
        st.success(f"Successfully loaded model from `{CHECKPOINT_FILE}`.")
    else:
        st.warning("⚠️ **DUMMY MODE:** Model file is a placeholder.")

    # --- Sidebar with "Golden" Timestamp Defaults ---
    with st.sidebar:
        st.header("Prediction Controls")

        # --- UPDATE THESE DEFAULTS with the output from your helper script ---
        default_date = datetime(2024, 6, 18)
        default_time_index = 31  # Corresponds to 15:30 UTC
        # --------------------------------------------------------------------

        time_options = [time(h, m) for h in range(24) for m in (0, 30)]
        selected_date = st.date_input("Select a Date", default_date)
        selected_time = st.selectbox(
            "Select a Time (UTC)", options=time_options, index=default_time_index
        )
        target_datetime = datetime.combine(selected_date, selected_time)

    if st.button("Generate Prediction", type="primary"):
        if model == "dummy":
            st.info(
                "This is a placeholder. Once the real model is trained, it will show a real prediction."
            )
        else:
            condition_tensor, last_input_img, ground_truth_img = (
                load_data_with_gaps_filled(target_datetime)
            )

            with st.spinner("Model is generating prediction..."):
                device = next(model.parameters()).device
                prediction = model.diffusion_model.sample(
                    condition=condition_tensor.to(device),
                    image_size=IMAGE_SIZE,
                    channels=len(CHANNELS),
                )
                predicted_image = denormalize(prediction.cpu().numpy().squeeze())

            st.success("Prediction complete!")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.header("Last Input")
                for i, c in enumerate(CHANNELS):
                    st.image(np.clip(last_input_img[i], 0, 1), caption=c, use_container_width=True)
            with col2:
                st.header("Prediction")
                for i, c in enumerate(CHANNELS):
                    st.image(np.clip(predicted_image[i], 0, 1), caption=c, use_container_width=True)
            with col3:
                st.header("Ground Truth")
                for i, c in enumerate(CHANNELS):
                    st.image(np.clip(ground_truth_img[i], 0, 1), caption=c, use_container_width=True)
