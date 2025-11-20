import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from PIL import Image
import torch
from models.lpips_scorer import LPIPSScorer

scorer = LPIPSScorer(device="cuda")

img1 = Image.new("RGB", (256, 256), color="red")
img2 = Image.new("RGB", (256, 256), color="blue")

print("same image:", scorer.distance(img1, img1))     # ~0.0
print("red vs blue:", scorer.distance(img1, img2))    # > 0.2 ish
