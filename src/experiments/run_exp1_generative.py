#!/usr/bin/env python
"""
Experiment 1: Generative Uncertainty (Seed Variance)

For each prompt:
  - Generate N images from a fixed model with different random seeds.
  - Compute pairwise CLIP similarities, LPIPS distance, and latent similarities between the images.
  - Aggregate into mean and variance as a measure of generative uncertainty.

Outputs:
  - A CSV file with one row per (prompt, model) containing:
      prompt_id, prompt_text, category, model, num_images,
      clip_mean_similarity, clip_variance_similarity, 
      lpips_mean_distance, lpips_variance_distance,
        latent_mean_cosine_similarity, latent_variance_cosine_similarity,
        latent_mean_dimension_variance
"""

import argparse
import json
# import os
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from dataclasses import asdict
from pathlib import Path
from typing import List, Dict

import pandas as pd
from tqdm import tqdm

# --- Project imports: adjust if your structure differs ---
from src.prompts.prompt_dataset import PromptDataset
from src.generation.generator import ImageGenerator
from src.uncertainty.generative import (
    compute_generative_uncertainty,
    GenerativeUncertaintyResult,
)
from src.uncertainty.latent import compute_latent_uncertainty
from models.clip_encoder import ClipScorer
from models.lpips_scorer import LPIPSScorer
from models.stable_diffusion import StableDiffusionModel
# from models.sdxl import SDXLModel
# from models.pixart_alpha import PixArtModel


# ------------------ Helpers ------------------ #

MODEL_REGISTRY = {
    "sd15": StableDiffusionModel,
    # "sdxl": SDXLModel,
    # "pixart": PixArtModel,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 1: Generative Uncertainty (seed variance)"
    )

    parser.add_argument(
        "--prompts",
        type=str,
        default="/u/cz5047/reas-llm-uq/t2i-unc/prompts/simple_prompts.jsonl",
        help="Path to prompts JSONL file.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/exp1_generative",
        help="Base directory to store outputs (images + CSV).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sd15",
        choices=list(MODEL_REGISTRY.keys()),
        help="Which T2I model to use.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=8,
        help="Number of images (seeds) to generate per prompt.",
    )
    parser.add_argument(
        "--seed_offset",
        type=int,
        default=0,
        help="Offset for random seeds (seed = seed_offset + i).",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="*",
        default=None,
        help=(
            "If provided, restrict to these prompt categories "
            "(e.g., --categories specific moderate). "
            "Otherwise, use all prompts."
        ),
    )
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="Optional cap on number of prompts (for quick debugging).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for CLIP + T2I models (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--csv_name",
        type=str,
        default="exp1_generative.csv",
        help="Name of CSV file to write in out_dir.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable image caching (regenerate all images).",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="results/generated_images/images",
        help="Directory containing cached images (organized by model).",
    )

    return parser.parse_args()


def make_seeds(num_images: int, offset: int) -> List[int]:
    return [offset + i for i in range(num_images)]


def filter_prompts(dataset: PromptDataset, categories: List[str] | None):
    if categories is None:
        return dataset.entries

    categories_set = set(categories)
    return [p for p in dataset.entries if p.category in categories_set]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def main():
    args = parse_args()

    # Load prompts
    prompts_ds = PromptDataset(args.prompts)
    selected_prompts = filter_prompts(prompts_ds, args.categories)

    if args.max_prompts is not None:
        selected_prompts = selected_prompts[: args.max_prompts]

    if len(selected_prompts) == 0:
        raise ValueError("No prompts selected. Check --prompts and --categories.")

    prompts_ds.print_prompts()

    # Set up output dir structure
    out_dir = ensure_dir(args.out_dir)
    metadata_dir = ensure_dir(out_dir / "metadata")

    # Save the run config for reproducibility
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Instantiate model + generator + CLIP scorer
    ModelCls = MODEL_REGISTRY[args.model]
    t2i_model = ModelCls(device=args.device)
    use_cache = not args.no_cache
    
    # Use cached images directory if available, otherwise use experiment-specific directory
    if use_cache and args.cache_dir:
        cache_path = Path(args.cache_dir) / t2i_model.name
        if cache_path.exists():
            images_dir = cache_path
            print(f"Using cached images from: {images_dir}")
        else:
            images_dir = ensure_dir(out_dir / "images")
            print(f"Cache directory {cache_path} not found, using: {images_dir}")
    else:
        images_dir = ensure_dir(out_dir / "images")
    
    generator = ImageGenerator(model=t2i_model, out_dir=str(images_dir), use_cache=use_cache)
    clip_scorer = ClipScorer(device=args.device)
    lpips_scorer = LPIPSScorer(device=args.device)

    results: List[Dict] = []

    # Loop over prompts
    print(
        f"Running generative uncertainty experiment with model={args.model}, "
        f"{len(selected_prompts)} prompts, {args.num_images} images/prompt."
    )

    return_latents = False

    for prompt_entry in tqdm(selected_prompts, desc="Prompts"):
        prompt_id = getattr(prompt_entry, "id", None) or prompt_entry.text[:32]
        prompt_text = prompt_entry.text
        category = getattr(prompt_entry, "category", "unknown")

        seeds = make_seeds(args.num_images, args.seed_offset)

        # Generate images for this prompt (uses caching if images already exist)
        # ImageGenerator returns list of dicts:
        # {"image": PIL.Image, "prompt": str, "seed": int, "latents": torch.Tensor (optional), "metadata": {...}}
        gen_results = generator.generate_for_prompt(
            prompt_text,
            seeds=seeds,
            prompt_id=prompt_id,
            extra_meta={
                "prompt_id": prompt_id,
                "category": category,
                "experiment": "exp1_generative",
            },
            return_latents=return_latents,
        )

        images = [r["image"] for r in gen_results]
        # Latents might not be present if they weren't cached and caching failed
        latents = [r.get("latents") for r in gen_results if "latents" in r]
        
        # Ensure we have latents for all images (if return_latents was True)
        if return_latents and len(latents) < len(images):
            print(f"Warning: Missing latents for {len(images) - len(latents)} images for prompt {prompt_id}")
            # If we're missing latents, we'll skip latent uncertainty computation
            latents = []

        meta_path = metadata_dir / f"{prompt_id}.json"
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "prompt_id": prompt_id,
                    "prompt_text": prompt_text,
                    "category": category,
                    "model": t2i_model.name,
                    "seeds": seeds,
                    "num_images": len(images),
                },
                f,
                indent=2,
            )

        # Compute generative uncertainty metrics for this prompt
        (clip_mean_sim, clip_var_sim), (lpips_mean_sim, lpips_var_sim) = compute_generative_uncertainty(
            clip_scorer, lpips_scorer, images
        )

        # Compute latent uncertainty if we have latents
        if len(latents) >= 2:
            latent_unc = compute_latent_uncertainty(latents)
        else:
            # Use default values if latents unavailable
            latent_unc = type('obj', (object,), {
                'mean_cosine_similarity': 0.0,
                'var_cosine_similarity': 0.0,
                'mean_dim_variance': 0.0
            })()

        res = GenerativeUncertaintyResult(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            category=category,
            model=t2i_model.name,
            num_images=len(images),
            clip_mean_similarity=clip_mean_sim,
            clip_variance_similarity=clip_var_sim,
            lpips_mean_distance=lpips_mean_sim,
            lpips_variance_distance=lpips_var_sim,
            latent_mean_cosine_similarity=latent_unc.mean_cosine_similarity,
            latent_variance_cosine_similarity=latent_unc.var_cosine_similarity,
            latent_mean_dimension_variance=latent_unc.mean_dim_variance,
        )

        results.append(asdict(res))

    # Save results as CSV
    df = pd.DataFrame(results)
    csv_path = out_dir / args.csv_name
    df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")


if __name__ == "__main__":
    main()
