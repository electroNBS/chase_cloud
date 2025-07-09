# src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- Reusable Building Block ---
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2 + time embedding"""

    def __init__(self, in_channels, out_channels, time_emb_dim, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        # Time embedding projection
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, t):
        h = self.double_conv(x)
        
        # Project time embedding and add it
        time_emb = F.relu(self.time_mlp(t))
        # Expand time_emb to match spatial dimensions of h
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h.shape[2], h.shape[3])
        
        return h + time_emb

# --- Downsampling Block for UNet Encoder ---
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, time_emb_dim)
        )

    def forward(self, x, t):
        # We need to pass 't' to the DoubleConv block
        # Since Sequential doesn't support multiple args, we call it directly
        for layer in self.maxpool_conv:
            if isinstance(layer, DoubleConv):
                x = layer(x, t)
            else:
                x = layer(x)
        return x

# --- Upsampling Block for UNet Decoder ---
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # The input to DoubleConv here is `in_channels` because we concatenate
        # the skip connection (in_channels//2) and the upsampled tensor (in_channels//2)
        self.conv = DoubleConv(in_channels, out_channels, time_emb_dim)

    def forward(self, x1, x2, t):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, t)

class SinusoidalPositionEmbeddings(nn.Module):
    # (This class remains unchanged)
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# --- The Main Refactored UNet Model ---
class ConditionalUNet(nn.Module):
    def __init__(self, image_size, in_channels, out_channels, condition_channels, time_emb_dim=256):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        total_in_channels = in_channels + condition_channels

        # --- Encoder Path (pass time_emb_dim to each block) ---
        self.inc = DoubleConv(total_in_channels, 64, time_emb_dim)
        self.down1 = Down(64, 128, time_emb_dim)
        self.down2 = Down(128, 256, time_emb_dim)
        self.down3 = Down(256, 512, time_emb_dim)
        self.down4 = Down(512, 1024, time_emb_dim)
        
        # --- Decoder Path (pass time_emb_dim to each block) ---
        self.up1 = Up(1024, 512, time_emb_dim)
        self.up2 = Up(512, 256, time_emb_dim)
        self.up3 = Up(256, 128, time_emb_dim)
        self.up4 = Up(128, 64, time_emb_dim)
        
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, timestep, condition):
        x = torch.cat([x, condition], dim=1)
        t = self.time_mlp(timestep)

        # --- THIS IS THE CORRECTED LOGIC ---
        # Pass `t` to each block
        x1 = self.inc(x, t)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x4 = self.down3(x3, t)
        x5 = self.down4(x4, t)
        
        # The erroneous addition at the bottleneck is now removed.
        # Time is handled inside each block.
        
        x = self.up1(x5, x4, t)
        x = self.up2(x, x3, t)
        x = self.up3(x, x2, t)
        x = self.up4(x, x1, t)
        
        logits = self.outc(x)
        return logits

# --- The CloudDiffusion class remains the same ---
class CloudDiffusion(nn.Module):
    # This class does not need any changes
    def __init__(self, model, timesteps, beta_start, beta_end):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        
    def _get_index_from_list(self, vals, t, x_shape):
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
        
    def forward_diffusion(self, x0, t, noise):
        alpha_t = self._get_index_from_list(self.alphas_cumprod, t, x0.shape)
        xt = torch.sqrt(alpha_t) * x0 + torch.sqrt(1. - alpha_t) * noise
        return xt

    def forward(self, x0, condition):
        t = torch.randint(0, self.timesteps, (x0.shape[0],), device=x0.device).long()
        noise = torch.randn_like(x0)
        xt = self.forward_diffusion(x0, t, noise)
        predicted_noise = self.model(xt, t, condition)
        loss = F.l1_loss(noise, predicted_noise)
        return loss

    @torch.no_grad()
    def sample(self, condition, image_size, channels):
        device = next(self.model.parameters()).device
        shape = (condition.shape[0], channels, image_size, image_size)
        img = torch.randn(shape, device=device)
        
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            predicted_noise = self.model(img, t, condition)
            
            alpha = self.alphas[i]
            alpha_cumprod_t = self.alphas_cumprod[i]
            
            if i > 0:
                noise = torch.randn_like(img)
                beta_t = self.betas[i]
                sigma_t = torch.sqrt(beta_t)
            else:
                noise = torch.zeros_like(img)
                sigma_t = 0
            
            term1 = 1 / torch.sqrt(alpha)
            term2 = (1 - alpha) / (torch.sqrt(1 - alpha_cumprod_t))
            img = term1 * (img - term2 * predicted_noise) + sigma_t * noise
        return img