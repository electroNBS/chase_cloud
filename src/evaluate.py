# src/evaluate.py

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from skimage.metrics import (
    structural_similarity as ssim,
    peak_signal_noise_ratio as psnr,
)
import numpy as np

from .train import CloudChaseModule  # Import the main module
from .data_loader import INSATCloudDataset
from .utils import denormalize
from .config import *


def evaluate_prediction(prediction, ground_truth):
    pred_norm = denormalize(prediction)
    gt_norm = denormalize(ground_truth)

    mae = np.mean(np.abs(pred_norm - gt_norm))
    ssim_scores, psnr_scores = [], []
    for i in range(pred_norm.shape[0]):  # Iterate over channels
        ssim_val = ssim(gt_norm[i], pred_norm[i], data_range=1.0)
        psnr_val = psnr(gt_norm[i], pred_norm[i], data_range=1.0)
        ssim_scores.append(ssim_val)
        psnr_scores.append(psnr_val)

    return {"MAE": mae, "SSIM": np.mean(ssim_scores), "PSNR": np.mean(psnr_scores)}


def generate_and_visualize(model, dataloader, device):
    model.eval()
    model.to(device)

    condition, ground_truth = next(iter(dataloader))
    condition_sample = condition[0:1].to(device)
    ground_truth_sample = ground_truth[0:1].cpu().numpy()

    print("Generating prediction...")
    with torch.no_grad():
        prediction = model.diffusion_model.sample(
            condition=condition_sample, image_size=IMAGE_SIZE, channels=len(CHANNELS)
        )
    prediction_sample = prediction.cpu().numpy()
    print("Generation complete.")

    # --- Visualization ---
    pred_viz = denormalize(prediction_sample[0])
    gt_viz = denormalize(ground_truth_sample[0])
    last_input_frame = denormalize(condition[0, -len(CHANNELS) :, :, :].cpu().numpy())

    fig, axes = plt.subplots(len(CHANNELS), 3, figsize=(12, 4 * len(CHANNELS)))
    fig.suptitle("Cloud Motion Prediction", fontsize=16)

    for i, channel_name in enumerate(CHANNELS):
        axes[i, 0].imshow(last_input_frame[i], cmap="gray")
        axes[i, 0].set_title(f"Last Input ({channel_name})")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt_viz[i], cmap="gray")
        axes[i, 1].set_title(f"Ground Truth ({channel_name})")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_viz[i], cmap="gray")
        axes[i, 2].set_title(f"Prediction ({channel_name})")
        axes[i, 2].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("outputs/figures/prediction_comparison.png")
    plt.show()

    # --- Quantitative Evaluation ---
    metrics = evaluate_prediction(prediction_sample[0], ground_truth_sample[0])
    print("\n--- Evaluation Metrics ---")
    for key, val in metrics.items():
        print(f"{key}: {val:.4f}")
    print("------------------------")


if __name__ == "__main__":
    # Path to your saved model checkpoint
    # Find the best checkpoint in your 'checkpoints' folder
    checkpoint_file = "D:\chase-the-cloud\checkpoints\cloud-chase-best-epoch=22-train_loss=0.80.ckpt"  # Example name

    # Load the model from the checkpoint
    model = CloudChaseModule.load_from_checkpoint(checkpoint_file)

    # Create a dataloader for the test/validation set
    # It's good practice to have a separate test set folder
    test_dataset = INSATCloudDataset(
        root_dir=DATA_PATH,
        channels=CHANNELS,
        input_frames=INPUT_FRAMES,
        future_frames=FUTURE_FRAMES,
        image_size=IMAGE_SIZE,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generate_and_visualize(model, test_loader, device)
