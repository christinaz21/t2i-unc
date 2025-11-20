from __future__ import annotations

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)


from models.stable_diffusion import StableDiffusionModel
from PIL import Image
import numpy as np
import torch
from PIL import Image


model = StableDiffusionModel(device="cuda")

out = model.generate("a red apple on a table", num_images=2, seeds=[123, 123], return_latents=True)

out_dir = "outputs"
os.makedirs(out_dir, exist_ok=True)

img = out[0]["image"]

if hasattr(img, "save"):
    path = os.path.join(out_dir, "image_0.png")
    img.save(path)
else:
    arr = img if isinstance(img, np.ndarray) else np.array(img)
    # normalize if not uint8
    if arr.dtype != np.uint8:
        arr = (255 * (arr - arr.min()) / (arr.max() - arr.min())).astype("uint8")
    path = os.path.join(out_dir, "image_0.png")
    Image.fromarray(arr).save(path)

print(f"Saved {path}")


import torch
from torchvision.utils import save_image  # pip install torchvision

def save_latent_as_grid(latents: torch.Tensor, path: str):
    """
    latents: (4, H, W) or (1,4,H,W) tensor
    Saves a 2x2 grid of the 4 channels as grayscale.
    """
    if latents.ndim == 3:
        latents = latents.unsqueeze(0)  # (1,4,H,W)
    # (1,4,H,W) -> (4,1,H,W) each channel as one image
    chs = latents[0].unsqueeze(1)  # (4,1,H,W)

    # Normalize per-channel to [0,1] for visualization
    chs = (chs - chs.min()) / (chs.max() - chs.min() + 1e-8)

    # save_image will arrange them into a grid
    save_image(chs, path, nrow=2)

latents = out[0]["latents"]  # (4,H,W) tensor
latent_path = os.path.join(out_dir, "latents_0.png")
save_latent_as_grid(latents, latent_path)
print(f"Saved latent visualization {latent_path}")


def latent_to_pca_rgb(latents: torch.Tensor) -> torch.Tensor:
    """
    Project latent channels onto 3 principal components and return an
    RGB tensor of shape (3, H, W) in [0, 1].

    Args:
        latents: (C, H, W) or (1, C, H, W) tensor.

    Returns:
        rgb: torch.Tensor of shape (3, H, W), dtype float32, in [0, 1].
    """

    latents = latents.float()

    # Ensure shape (C, H, W)
    if latents.ndim == 4:
        # (1, C, H, W) -> (C, H, W)
        latents = latents[0]
    assert latents.ndim == 3, f"Expected (C,H,W) or (1,C,H,W), got {latents.shape}"

    C, H, W = latents.shape

    # Flatten spatial dims: X has shape (N, C), where N=H*W
    # Each row is a spatial position; features are channels
    X = latents.view(C, -1).T  # (N, C)

    # Center data
    X_mean = X.mean(dim=0, keepdim=True)  # (1, C)
    X_centered = X - X_mean

    # Compute PCA via SVD on centered data
    # X_centered: (N, C) -> U: (N, C), S: (C,), Vh: (C, C)
    # Columns of V = PCs in channel space
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    V = Vh.T  # (C, C)

    # Take top 3 principal directions in channel space
    W_pca = V[:, :3]  # (C, 3)

    # Project N x C → N x 3
    X_proj = X_centered @ W_pca  # (N, 3)

    # Reshape back to (3, H, W)
    rgb = X_proj.T.view(3, H, W)  # (3, H, W)

    # Normalize each channel independently to [0,1] for visualization
    # Avoid degenerate constant channels:
    for c in range(3):
        ch = rgb[c]
        min_val = ch.min()
        max_val = ch.max()
        denom = (max_val - min_val).clamp(min=1e-8)
        rgb[c] = (ch - min_val) / denom

    return rgb.float()



def save_latent_pca_image(latents: torch.Tensor, path: str) -> None:
    """
    Convenience wrapper: compute PCA RGB projection and save to disk.

    Args:
        latents: (C,H,W) or (1,C,H,W) tensor (on any device).
        path: output PNG path.
    """
    # Make sure we're on CPU
    latents_cpu = latents.detach().cpu()
    rgb = latent_to_pca_rgb(latents_cpu)  # (3,H,W) in [0,1]

    # Convert to HWC uint8 for PIL
    rgb_np = (rgb.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype("uint8")
    img = Image.fromarray(rgb_np)
    img.save(path)

save_latent_pca_image(latents, os.path.join(out_dir, "latents_pca_0.png"))



