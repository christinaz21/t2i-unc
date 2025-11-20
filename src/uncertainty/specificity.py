from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class SpecificityResult:
    base_id: str     # logical group of prompts (specific/moderate/abstract)
    model: str
    sim_specific_abstract: float
    sim_specific_moderate: float

def compute_specificity_scores(clip_scorer, img_specific, img_abstract, img_moderate):
    s_a = clip_scorer.image_image_similarity(img_specific, img_abstract)
    s_m = clip_scorer.image_image_similarity(img_specific, img_moderate)
    return s_a, s_m
