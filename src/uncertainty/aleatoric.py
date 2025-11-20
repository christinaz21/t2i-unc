# src/uncertainty/aleatoric.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict

from PIL import Image

from models.clip_encoder import ClipScorer


@dataclass
class AleatoricUncertaintyResult:
    """
    Aleatoric uncertainty metrics for a single (prompt, seed, image).
    """
    prompt_id: str
    prompt_text: str
    category: str
    model: str
    seed: int

    caption: str

    sim_prompt_image: float
    sim_caption_image: float
    sim_prompt_caption: float

    # Derived metrics
    delta_image_sim: float      # sim_prompt_image - sim_caption_image
    aleatoric_prompt_caption: float  # 1 - sim_prompt_caption (higher = more uncertain)

    num_images: int

def compute_aleatoric_uncertainty_for_image(
    clip_scorer: ClipScorer,
    captioner,
    image: Image.Image,
    prompt_id: str,
    prompt_text: str,
    category: str,
    model_name: str,
    seed: int,
) -> AleatoricUncertaintyResult:
    """
    Compute aleatoric uncertainty metrics for a single image generated from a prompt.

    Args:
        clip_scorer: CLIP wrapper
        captioner: object with method `caption(image: PIL.Image) -> str`
        image: generated image (PIL)
        prompt_id: ID of the prompt
        prompt_text: original text prompt
        category: prompt category (specific/abstract/etc.)
        model_name: name of T2I model
        seed: RNG seed used for generation

    Returns:
        AleatoricSampleResult
    """
    # Caption the image with a VLM
    caption = captioner.caption(image)

    # CLIP similarities
    sim_prompt_image = clip_scorer.image_text_similarity(image, prompt_text)
    sim_caption_image = clip_scorer.image_text_similarity(image, caption)
    sim_prompt_caption = clip_scorer.text_text_similarity(prompt_text, caption)

    # Derived metrics
    delta_image_sim = sim_prompt_image - sim_caption_image
    aleatoric_prompt_caption = 1.0 - sim_prompt_caption  # higher = more mismatch

    return AleatoricUncertaintyResult(
        prompt_id=prompt_id,
        prompt_text=prompt_text,
        category=category,
        model=model_name,
        seed=seed,
        caption=caption,
        sim_prompt_image=sim_prompt_image,
        sim_caption_image=sim_caption_image,
        sim_prompt_caption=sim_prompt_caption,
        delta_image_sim=delta_image_sim,
        aleatoric_prompt_caption=aleatoric_prompt_caption,
        num_images=1,
    )

