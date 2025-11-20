import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)


from models.clip_encoder import ClipScorer
from PIL import Image
import numpy as np

clip = ClipScorer(device="cuda")

img = Image.new("RGB", (256, 256), color="red")

img_path = "/u/cz5047/reas-llm-uq/t2i-unc/outputs/image_0.png"
img = Image.open(img_path).convert("RGB")

good_caption = "a shiny red apple on a brown wooden table in front of a dark green wall"
bad_caption = "a blue car driving down the street on a sunny day"

print("img-text sim (good):", clip.image_text_similarity(img, good_caption))
print("img-img sim:", clip.image_image_similarity(img, img))

print("img-text sim (bad):", clip.image_text_similarity(img, bad_caption))


out_dir = "outputs"

if hasattr(img, "save"):
    path = os.path.join(out_dir, "apple.png")
    img.save(path)
else:
    arr = img if isinstance(img, np.ndarray) else np.array(img)
    # normalize if not uint8
    if arr.dtype != np.uint8:
        arr = (255 * (arr - arr.min()) / (arr.max() - arr.min())).astype("uint8")
    path = os.path.join(out_dir, "red.png")
    Image.fromarray(arr).save(path)

print(f"Saved {path}")