# Chase the Cloud: Cloud Motion Prediction with Diffusion Models

  <!-- Optional: Create a nice banner for your project -->

This project, "Chase the Cloud," leverages cutting-edge deep generative models to predict short-term cloud motion. By applying conditional Denoising Diffusion Probabilistic Models (DDPMs) to multi-spectral satellite imagery from India's indigenous **INSAT-3DR/3DS** satellites, we aim to provide high-fidelity, realistic forecasts of cloud evolution.

Traditional optical flow or physics-based models often falter when confronted with volatile and non-linear weather dynamics. This project uses a data-driven, spatio-temporal learning approach to simulate cloud evolution, delivering enhanced forecasting for nowcasting (0–3 hours) and early warnings for severe weather events like thunderstorms.

## ✨ Key Features

*   **Generative Forecasting:** Utilizes a conditional diffusion model to generate future cloud frames, capturing the complex textures and stochastic nature of cloud formation and dissipation.
*   **Multi-Spectral Input:** Fuses data from multiple satellite channels (Visible, Infrared, Water Vapor) to provide a richer context for more accurate predictions.
*   **Indigenous Data Focus:** Demonstrates the viability of using data from Indian satellites (INSAT-3DR/3DS) for frontier AI research in meteorology.
*   **Spatio-Temporal Learning:** The model is conditioned on a sequence of past frames (e.g., 4 frames over 2 hours) to predict future frames (e.g., 1-2 frames in the next hour).
*   **Modular & Scalable:** Built with PyTorch Lightning for clean, structured, and scalable code.

## 🎯 Project Objectives

1.  **Develop a Deep Generative Model:** Implement a conditional diffusion network for short-term cloud motion forecasting.
2.  **Utilize INSAT Imagery:** Process and train the model on multi-channel Level-1C data from INSAT-3DR/3DS.
3.  **Improve Nowcasting:** Create a prototype system for improved nowcasting, with a focus on severe weather applications.
4.  **Evaluate Performance:** Quantitatively and qualitatively assess the model's predictions against ground truth data using metrics like SSIM, PSNR, and MAE.

## 🛠️ Tech Stack & Architecture

*   **Backend & Deep Learning:** Python, PyTorch, PyTorch Lightning
*   **Geospatial Data Handling:** `rioxarray`, `gdal`, `xarray`, `opencv-python`
*   **Model Architecture:** Conditional UNet backbone within a Denoising Diffusion Probabilistic Model (DDPM) framework.
*   **Data Processing:** Custom data loaders handle the specific `.tif` format of INSAT data, including aspect-ratio-preserving resizing and normalization.
*   **Visualization:** `matplotlib`, `plotly`
*   **Prototyping:** `streamlit` (for the planned GUI)

<p align="center">
  <img src=https://github.com/user-attachments/assets/c2bf7fcd-c8cd-4073-855e-901197592e83 width="700"> <!-- Optional: Create a simple architecture diagram -->
  <br>
  <em>High-level architecture of the conditional diffusion model for cloud forecasting.</em>
</p>

## 📂 Project Structure

The repository is organized to separate concerns, making it easy to navigate and extend. <br>
chase-the-cloud/ <br>
├── data/raw_insat/ &nbsp;&nbsp;                  # Place your raw .tif satellite images here <br>
├── checkpoints/   &nbsp;&nbsp;                  # Saved model weights will appear here <br>
├── outputs/             &nbsp;&nbsp;            # Generated figures and metrics are saved here <br>
├── src/                 &nbsp;&nbsp;            # All source code <br>
│ ├── config.py           &nbsp;&nbsp;           # Central configuration for hyperparameters and paths <br>
│ ├── data_loader.py      &nbsp;&nbsp;           # Custom PyTorch Dataset for INSAT data <br>
│ ├── model.py          &nbsp;&nbsp;             # UNet and Diffusion model implementations <br>
│ ├── train.py         &nbsp;&nbsp;              # Main script to start model training <br>
│ ├── evaluate.py       &nbsp;&nbsp;             # Script to evaluate a trained model <br>
│ └── utils.py         &nbsp;&nbsp;              # Helper functions (normalization, resizing) <br>
├── app/                &nbsp;&nbsp;             # Code for the Streamlit GUI prototype <br>
├── .gitignore <br>
├── README.md <br>
└── requirements.txt <br>
 <br>
## 🚀 Getting Started

### 1. Prerequisites

*   Python 3.9+ <br>
*   NVIDIA GPU with CUDA support (highly recommended for training) <br>
*   Access to INSAT-3DR/3DS Level-1C data in `.tif` format. <br>

### 2. Clone the Repository

```bash
git clone https://github.com/your-username/chase-the-cloud.git
cd chase-the-cloud
```
### 3. Set Up the Environment
It is highly recommended to use a virtual environment. <br>
# Create and activate a conda environment
```
conda create -n cloud_chase python=3.9
conda activate cloud_chase
```
# Install all required packages
```
pip install -r requirements.txt
```

### 4. Prepare the Data
Place all your INSAT .tif files inside the data/raw_insat/ directory. The data loader expects filenames in the following format:  <br>
`3DIMG_DDMMMYYYY_HHMM_..._IMG_CHANNEL.tif` (e.g., 3DIMG_18JUN2024_0030_..._IMG_MIR.tif). Files are downloaded in this format from ISRO's MOSDAC website. <br>

### 5. Configure the Model
Adjust the parameters in src/config.py to suit your needs. You can change the image size, batch size, learning rate, and input channels. <br>
# src/config.py
```
IMAGE_SIZE = 256
BATCH_SIZE = 8
CHANNELS = ['MIR', 'WV', 'VIS'] # Ensure these match your data
```

### 6. Train the Model
Start the training process by running the train.py script from the root directory. <br>
```
python src/train.py
```
Training progress and logs will be displayed in the console. The best model checkpoint will be saved automatically in the checkpoints/ directory. <br>

### 7. Evaluate a Trained Model
After training is complete, you can generate predictions and evaluate the model's performance. First, update the checkpoint_file path in src/evaluate.py to point to your saved model. <br>
# src/evaluate.py
```
checkpoint_file = "checkpoints/cloud-chase-best-....ckpt" # Update this
```
Then, run the evaluation script: <br>
```
python src/evaluate.py
```
This will generate a comparison plot in outputs/figures/ and print quantitative metrics (SSIM, PSNR, MAE) to the console. <br>
### 📈 Expected Outcomes:
- Trained Model: A robust diffusion model capable of generating realistic 1-2 frame future cloud predictions. <br>
- Research Insights: A deeper understanding of applying generative models to geospatial and meteorological data. <br>
- Prototype Interface: A simple GUI (planned for app/) to visualize predictions. <br>
### Future enhancements could include:
- Implementing an autoregressive loop for multi-frame prediction (>2 frames). <br>
- Developing a patch-based training and inference pipeline to work with full-resolution imagery. <br>
- Integrating perceptual and adversarial losses to improve visual fidelity. <br>
- Deploying the model as a REST API for integration with operational nowcasting systems. <br>
