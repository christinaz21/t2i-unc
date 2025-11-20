from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class GenerativeUncertaintyResult:
    prompt_id: str
    prompt_text: str
    model: str
    clip_mean_similarity: float
    clip_variance_similarity: float
    lpips_mean_distance: float
    lpips_variance_distance: float
    category: str
    num_images: int
    latent_mean_cosine_similarity: float
    latent_variance_cosine_similarity: float
    latent_mean_dimension_variance: float

def compute_generative_uncertainty(clip_scorer, lpips_scorer, images: List):
    # pairwise CLIP similarities
    clip_sims = []
    lpips_sims = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            clip_sims.append(clip_scorer.image_image_similarity(images[i], images[j]))
            lpips_sims.append(lpips_scorer.distance(images[i], images[j]))
    clip_sims = np.array(clip_sims)
    lpips_sims = np.array(lpips_sims)
    return (clip_sims.mean(), clip_sims.var()), (lpips_sims.mean(), lpips_sims.var())
