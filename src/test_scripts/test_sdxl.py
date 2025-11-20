from __future__ import annotations

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)


from models.sdxl import SDXLModel
from PIL import Image
import numpy as np
import torch
from PIL import Image


model = SDXLModel(device="cuda")

out = model.generate("a red apple on a table", num_images=2, seeds=[123, 123])

out_dir = "outputs"
os.makedirs(out_dir, exist_ok=True)

img = out[0]["image"]

if hasattr(img, "save"):
    path = os.path.join(out_dir, "sdxl_image_0.png")
    img.save(path)
else:
    arr = img if isinstance(img, np.ndarray) else np.array(img)
    # normalize if not uint8
    if arr.dtype != np.uint8:
        arr = (255 * (arr - arr.min()) / (arr.max() - arr.min())).astype("uint8")
    path = os.path.join(out_dir, "sdxl_image_0.png")
    Image.fromarray(arr).save(path)

print(f"Saved {path}")




