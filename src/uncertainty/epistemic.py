from dataclasses import dataclass
from typing import List
import numpy as np
import PIL


@dataclass
class EpistemicResult:
    prompt_id: str
    prompt_text: str
    models: list[str]
    mean_similarity: float
    variance_similarity: float

def compute_epistemic_uncertainty(clip_scorer, images_per_model: dict[str, "PIL.Image"]):
    sims = []
    models = list(images_per_model.keys())
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            sims.append(
                clip_scorer.image_image_similarity(
                    images_per_model[models[i]],
                    images_per_model[models[j]],
                )
            )
    sims = np.array(sims)
    return sims.mean(), sims.var()
