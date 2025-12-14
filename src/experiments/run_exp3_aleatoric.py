# src/experiments/run_exp3_aleatoric.py

#!/usr/bin/env python

"""
Experiment 3: Aleatoric Uncertainty (Prompt-Caption Misalignment)

For each prompt and seed:
  - Generate an image with a T2I model.
  - Caption the image using a VLM.
  - Compute CLIP-based similarities:
      sim(prompt, image), sim(caption, image), sim(prompt, caption)
  - Derive aleatoric metrics = how much the caption diverges from the prompt.

Outputs:
  - CSV with one row per (prompt, seed, model)
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict

import pandas as pd
from tqdm import tqdm

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from src.prompts.prompt_dataset import PromptDataset
from src.generation.generator import ImageGenerator
from models.clip_encoder import ClipScorer
from models.vlm_captioner import Captioner
from src.uncertainty.aleatoric import compute_aleatoric_uncertainty_for_image, AleatoricUncertaintyResult
from models.stable_diffusion import StableDiffusionModel
# from models.sdxl import SDXLModel
# from models.pixart_alpha import PixArtModel

import numpy as np

MODEL_REGISTRY = {
    "sd15": StableDiffusionModel,
    # "sdxl": SDXLModel,
    # "pixart": PixArtModel,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Experiment 3: Aleatoric Uncertainty via prompt-caption alignment"
    )
    p.add_argument(
        "--prompts",
        type=str,
        default="/u/cz5047/reas-llm-uq/t2i-unc/prompts/simple_prompts.jsonl",
        help="Path to prompts JSONL file.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="results/exp3_aleatoric",
        help="Directory to store CSV and metadata.",
    )
    p.add_argument(
        "--model",
        type=str,
        default="sd15",
        choices=list(MODEL_REGISTRY.keys()),
        help="Which T2I model to use.",
    )
    p.add_argument(
        "--num_images",
        type=int,
        default=4,
        help="Number of images/seeds per prompt.",
    )
    p.add_argument(
        "--seed_offset",
        type=int,
        default=0,
        help="Seed offset (seed = seed_offset + i).",
    )
    p.add_argument(
        "--categories",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of prompt categories to include.",
    )
    p.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="Optional cap on number of prompts for quick runs.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for CLIP, VLM, and T2I ('cuda' or 'cpu').",
    )
    p.add_argument(
        "--csv_name",
        type=str,
        default="exp3_aleatoric.csv",
        help="Output CSV filename.",
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable image caching (regenerate all images).",
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        default="results/generated_images/images",
        help="Directory containing cached images (organized by model).",
    )
    return p.parse_args()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def make_seeds(num_images: int, offset: int) -> List[int]:
    return [offset + i for i in range(num_images)]


def filter_prompts(ds: PromptDataset, categories: List[str] | None):
    if categories is None:
        return ds.entries
    cats = set(categories)
    return [p for p in ds.entries if p.category in cats]


def main():
    args = parse_args()

    prompts_ds = PromptDataset(args.prompts)
    selected_prompts = filter_prompts(prompts_ds, args.categories)
    if args.max_prompts is not None:
        selected_prompts = selected_prompts[: args.max_prompts]

    if not selected_prompts:
        raise ValueError("No prompts selected. Check --prompts or --categories.")

    out_dir = ensure_dir(args.out_dir)
    meta_dir = ensure_dir(out_dir / "metadata")

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Instantiate models
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
    captioner = Captioner(device=args.device)   # you implement this

    results: List[Dict] = []

    print(
        f"Running aleatoric experiment with model={args.model}, "
        f"{len(selected_prompts)} prompts, {args.num_images} images/prompt."
    )

    for prompt_entry in tqdm(selected_prompts, desc="Prompts"):
        prompt_id = getattr(prompt_entry, "id", None) or prompt_entry.text[:32]
        prompt_text = prompt_entry.text
        category = getattr(prompt_entry, "category", "unknown")

        seeds = make_seeds(args.num_images, args.seed_offset)

        gen_results = generator.generate_for_prompt(
            prompt_text,
            seeds=seeds,
            prompt_id=prompt_id,
            extra_meta={
                "prompt_id": prompt_id,
                "category": category,
                "experiment": "exp3_aleatoric",
            },
            return_latents=False,
        )

        images = [r["image"] for r in gen_results]
        image_stats: List[dict] = []
        sims_prompt_image = []
        sims_caption_image = []
        sims_prompt_caption = []
        delta_image_sims = []
        aleatoric_prompt_captions = []

        # One aleatoric row per generated image/seed
        for r in gen_results:
            image = r["image"]
            seed = r["seed"]

            alea_res = compute_aleatoric_uncertainty_for_image(
                clip_scorer=clip_scorer,
                captioner=captioner,
                image=image,
                prompt_id=prompt_id,
                prompt_text=prompt_text,
                category=category,
                model_name=t2i_model.name,
                seed=seed,
            )

            stats = asdict(alea_res)
            image_stats.append(stats)

            sims_prompt_image.append(stats["sim_prompt_image"])
            sims_caption_image.append(stats["sim_caption_image"])
            sims_prompt_caption.append(stats["sim_prompt_caption"])
            delta_image_sims.append(stats["delta_image_sim"])
            aleatoric_prompt_captions.append(stats["aleatoric_prompt_caption"])


        image_stats = np.array(image_stats)

        # Optional: save per-prompt metadata
        meta_path = meta_dir / f"{prompt_id}.json"
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

        res = AleatoricUncertaintyResult(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            category=category,
            model=t2i_model.name,
            num_images=len(images),
            sim_prompt_image=float(np.mean(sims_prompt_image)),
            sim_caption_image=float(np.mean(sims_caption_image)),
            sim_prompt_caption=float(np.mean(sims_prompt_caption)),
            delta_image_sim=float(np.mean(delta_image_sims)),
            aleatoric_prompt_caption=float(np.mean(aleatoric_prompt_captions)),
            seed=-1,  # N/A at prompt level
            caption="",  # N/A at prompt level


        )

        results.append(asdict(res))

    df = pd.DataFrame(results)
    csv_path = out_dir / args.csv_name
    df.to_csv(csv_path, index=False)
    print(f"Saved aleatoric results to {csv_path}")


if __name__ == "__main__":
    main()
