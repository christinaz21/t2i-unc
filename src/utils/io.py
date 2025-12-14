# src/utils/io.py

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
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


def get_image_filename(
    prompt_id: str,
    seed: int,
    model: str,
    out_dir: Path | str,
) -> Path:
    """
    Generate a deterministic filename for an image based on prompt_id, seed, and model.
    
    Args:
        prompt_id: Unique identifier for the prompt
        seed: Random seed used for generation
        model: Model name (e.g., "sd15", "sdxl")
        out_dir: Directory where images are stored
        
    Returns:
        Path to the image file
    """
    out_dir = Path(out_dir)
    filename = f"{model}_{prompt_id}_seed{seed}.png"
    return out_dir / filename


def get_latent_filename(
    prompt_id: str,
    seed: int,
    model: str,
    out_dir: Path | str,
) -> Path:
    """
    Generate a deterministic filename for a latent tensor.
    
    Args:
        prompt_id: Unique identifier for the prompt
        seed: Random seed used for generation
        model: Model name (e.g., "sd15", "sdxl")
        out_dir: Directory where latents are stored
        
    Returns:
        Path to the latent file
    """
    out_dir = Path(out_dir)
    filename = f"{model}_{prompt_id}_seed{seed}.pt"
    return out_dir / filename


def load_cached_image(image_path: Path) -> Optional[Image.Image]:
    """
    Load a cached image if it exists.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        PIL Image if file exists, None otherwise
    """
    image_path = Path(image_path)
    if image_path.exists():
        try:
            return Image.open(image_path)
        except Exception as e:
            print(f"Warning: Failed to load cached image {image_path}: {e}")
            return None
    return None


def load_cached_latent(latent_path: Path):
    """
    Load a cached latent tensor if it exists.
    
    Args:
        latent_path: Path to the latent file
        
    Returns:
        torch.Tensor if file exists, None otherwise
    """
    import torch
    latent_path = Path(latent_path)
    if latent_path.exists():
        try:
            return torch.load(latent_path, map_location="cpu")
        except Exception as e:
            print(f"Warning: Failed to load cached latent {latent_path}: {e}")
            return None
    return None


def save_latent(latent, latent_path: Path):
    """
    Save a latent tensor to disk.
    
    Args:
        latent: torch.Tensor to save
        latent_path: Path where to save the latent
    """
    import torch
    latent_path = Path(latent_path)
    latent_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(latent, latent_path)


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
