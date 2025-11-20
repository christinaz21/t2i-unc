from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class GenerativeUncertaintyResult:
    prompt_id: str
    prompt_text: str
    model: str
    mean_similarity: float
    variance_similarity: float

def compute_generative_uncertainty(clip_scorer, images: List):
    # pairwise CLIP similarities
    sims = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            sims.append(clip_scorer.image_image_similarity(images[i], images[j]))
    sims = np.array(sims)
    return sims.mean(), sims.var()
