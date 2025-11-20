from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class AleatoricResult:
    prompt_id: str
    prompt_text: str
    model: str
    similarity_prompt_image: float
    similarity_prompt_caption: float
    semantic_drift: float  # e.g., 1 - similarity_prompt_caption

def compute_aleatoric_uncertainty(clip_scorer, captioner, image, prompt_text: str):
    caption = captioner.caption(image)
    sim_pi = clip_scorer.image_text_similarity(image, prompt_text)
    sim_pc = clip_scorer.image_text_similarity(image, caption)
    drift = 1 - sim_pc
    return sim_pi, sim_pc, drift, caption

