import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from PIL import Image
import torch
from models.vlm_captioner import Captioner

captioner = Captioner(device="cuda")

img1 = Image.new("RGB", (256, 256), color="red")
img2 = Image.new("RGB", (256, 256), color="blue")

img_apple = Image.open("/u/cz5047/reas-llm-uq/t2i-unc/outputs/apple.png").convert("RGB")
img_pixels = Image.open("/u/cz5047/reas-llm-uq/t2i-unc/outputs/latents_pca_0.png").convert("RGB")

print("same image:", captioner.caption(img1))
print("red vs blue:", captioner.caption(img2))
print("apple image:", captioner.caption(img_apple))
print("pixels image:", captioner.caption(img_pixels))