# src/utils/io.py

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import json
import re
import uuid

from PIL import Image  # make sure pillow is installed


def _slugify(text: str, max_len: int = 50) -> str:
    """
    Turn arbitrary text (like a prompt) into a safe, short filename fragment.
    """
    if not text:
        return ""
    # Lowercase, replace non-alphanumeric with underscores
    text = text.lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "", text)
    return text[:max_len] or ""


def save_image_with_metadata(
    image: Image.Image,
    out_dir: str | Path,
    metadata: Dict[str, Any],
) -> str:
    """
    Save an image and a JSON metadata file next to it.

    Filename pattern (roughly):
        {model}_{seed}_{prompt_slug}.png / .json

    Returns:
        str: path to the saved image.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = str(metadata.get("model", "model"))
    seed = metadata.get("seed")
    prompt = str(metadata.get("prompt", ""))

    seed_part = f"seed{seed}" if seed is not None else ""
    prompt_slug = _slugify(prompt)

    stem_parts = [model, prompt_slug, seed_part]
    stem_parts = [p for p in stem_parts if p]  # drop empties
    if stem_parts:
        stem = "_".join(stem_parts)
    else:
        stem = str(uuid.uuid4())

    img_path = out_dir / f"{stem}.png"
    base_stem = stem
    counter = 1

    # Avoid overwriting if same metadata appears multiple times
    while img_path.exists():
        stem = f"{base_stem}_{counter}"
        img_path = out_dir / f"{stem}.png"
        counter += 1

    # Save image
    image.save(img_path)

    # Save metadata JSON alongside
    meta_path = out_dir / f"{stem}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return str(img_path)
