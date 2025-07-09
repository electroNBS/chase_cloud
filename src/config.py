# src/config.py

# --- Data and Paths ---
DATA_PATH = "data/raw_insat"  # Relative path to your raw TIFF files
CHECKPOINT_PATH = "checkpoints/"  # Directory to save model weights

# --- Model & Data Parameters ---
# Match these with the channels in your filenames (e.g., MIR, WV, VIS)
CHANNELS = ["MIR", "WV", "VIS", "TIR1", "TIR2", "SWIR"]
IMAGE_SIZE = 128  # Start with 128 for faster training, then increase to 256
INPUT_FRAMES = 4  # Number of past frames to use as input
FUTURE_FRAMES = (
    2  # Number of future frames to predict (we only use the first one as target)
)

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 8  # Adjust based on your GPU memory
EPOCHS = 100  # Number of training epochs

# --- Diffusion Parameters ---
TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02
