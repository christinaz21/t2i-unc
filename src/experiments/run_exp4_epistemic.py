#!/usr/bin/env python
"""
Experiment 4: Epistemic Uncertainty (Cross-Model Comparison)

For each prompt:
  - Generate one image from each model (SD15, SDXL, PixArt) with the same seed.
  - Compute pairwise CLIP similarities between images from different models.
  - Aggregate into mean and variance as a measure of epistemic uncertainty.
  - Higher variance = more epistemic uncertainty (models disagree more).

Outputs:
  - A CSV file with one row per (prompt, models) containing:
      prompt_id, prompt_text, category, models, mean_similarity, variance_similarity
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict

import pandas as pd
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from dataclasses import asdict
from src.prompts.prompt_dataset import PromptDataset
from src.generation.generator import ImageGenerator
from src.uncertainty.epistemic import compute_epistemic_uncertainty, EpistemicResult
from models.clip_encoder import ClipScorer
from models.stable_diffusion import StableDiffusionModel
from models.sdxl import SDXLModel
from models.pixart_alpha import PixArtModel


MODEL_REGISTRY = {
    "sd15": StableDiffusionModel,
    "sdxl": SDXLModel,
    "pixart": PixArtModel,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 4: Epistemic Uncertainty (cross-model comparison)"
    )

    parser.add_argument(
        "--prompts",
        type=str,
        default="/u/cz5047/reas-llm-uq/t2i-unc/prompts/simple_prompts.jsonl",
        help="Path to prompts JSONL file.",
    )
    
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default="/u/cz5047/reas-llm-uq/t2i-unc/prompts",
        help="Directory containing prompt files (if using --all_categories).",
    )
    
    parser.add_argument(
        "--all_categories",
        action="store_true",
        help="Run on all prompt categories (abstract, concrete, moderate, underspecified).",
    )
    
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/exp4_epistemic",
        help="Base directory to store outputs (images + CSV).",
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["sd15", "sdxl", "pixart"],
        choices=list(MODEL_REGISTRY.keys()),
        help="Which T2I models to use for comparison (at least 2 required).",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Fixed seed to use for all models (ensures fair comparison). Default: 0 (matches generate_all_images.py).",
    )
    
    parser.add_argument(
        "--categories",
        type=str,
        nargs="*",
        default=None,
        help=(
            "If provided, restrict to these prompt categories "
            "(e.g., --categories abstract concrete). "
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
        default="exp4_epistemic.csv",
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


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def filter_prompts(dataset: PromptDataset, categories: List[str] | None):
    if categories is None:
        return dataset.entries
    categories_set = set(categories)
    return [p for p in dataset.entries if p.category in categories_set]


def main():
    args = parse_args()
    
    if len(args.models) < 2:
        raise ValueError("At least 2 models required for epistemic uncertainty comparison.")

    # Load prompts
    if args.all_categories:
        # Load from all category files
        prompt_dir = Path(args.prompt_dir)
        all_prompts = []
        category_files = ["abstract.jsonl", "concrete.jsonl", "moderate.jsonl", "underspecified.jsonl"]
        
        for cat_file in category_files:
            cat_path = prompt_dir / cat_file
            if cat_path.exists():
                dataset = PromptDataset(str(cat_path))
                all_prompts.extend(dataset.entries)
                print(f"Loaded {len(dataset.entries)} prompts from {cat_file}")
    else:
        # Load from single file
        prompts_ds = PromptDataset(args.prompts)
        all_prompts = filter_prompts(prompts_ds, args.categories)
    
    if args.max_prompts is not None:
        all_prompts = all_prompts[: args.max_prompts]

    if len(all_prompts) == 0:
        raise ValueError("No prompts selected. Check --prompts, --categories, or --all_categories.")

    print(f"\nTotal prompts to process: {len(all_prompts)}")
    print(f"Models to compare: {args.models}")
    print(f"Using fixed seed: {args.seed}")

    # Set up output dir structure
    out_dir = ensure_dir(args.out_dir)
    images_dir = ensure_dir(out_dir / "images")
    metadata_dir = ensure_dir(out_dir / "metadata")

    # Save the run config for reproducibility
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Instantiate all models
    models_dict = {}
    generators_dict = {}
    for model_name in args.models:
        ModelCls = MODEL_REGISTRY[model_name]
        model = ModelCls(device=args.device)
        models_dict[model_name] = model
        
        # Use cached images directory if available, otherwise use experiment-specific directory
        use_cache = not args.no_cache
        if use_cache and args.cache_dir:
            cache_path = Path(args.cache_dir) / model_name
            if cache_path.exists():
                model_images_dir = cache_path
                print(f"Using cached images for {model_name} from: {model_images_dir}")
            else:
                model_images_dir = ensure_dir(images_dir / model_name)
                print(f"Cache directory {cache_path} not found for {model_name}, using: {model_images_dir}")
        else:
            model_images_dir = ensure_dir(images_dir / model_name)
        
        generators_dict[model_name] = ImageGenerator(
            model=model, 
            out_dir=str(model_images_dir), 
            use_cache=use_cache
        )
        print(f"Loaded model: {model_name}")

    clip_scorer = ClipScorer(device=args.device)
    results: List[Dict] = []

    print(f"\nRunning epistemic uncertainty experiment...")
    print("=" * 80)

    # Loop over prompts
    for prompt_entry in tqdm(all_prompts, desc="Prompts"):
        prompt_id = getattr(prompt_entry, "id", None) or prompt_entry.text[:32]
        prompt_text = prompt_entry.text
        category = getattr(prompt_entry, "category", "unknown")

        # Generate one image per model with the same seed
        images_per_model = {}
        
        for model_name in args.models:
            generator = generators_dict[model_name]
            
            gen_results = generator.generate_for_prompt(
                prompt_text,
                seeds=[args.seed],
                prompt_id=prompt_id,
                extra_meta={
                    "prompt_id": prompt_id,
                    "category": category,
                    "experiment": "exp4_epistemic",
                    "model": model_name,
                },
                return_latents=False,
            )
            
            if len(gen_results) > 0:
                images_per_model[model_name] = gen_results[0]["image"]
            else:
                print(f"Warning: No image generated for {model_name}, prompt {prompt_id}")
                continue

        # Need at least 2 models to compute epistemic uncertainty
        if len(images_per_model) < 2:
            print(f"Warning: Skipping {prompt_id} - need at least 2 models, got {len(images_per_model)}")
            continue

        # Compute epistemic uncertainty (cross-model similarity)
        mean_sim, var_sim = compute_epistemic_uncertainty(clip_scorer, images_per_model)

        # Save metadata
        meta_path = metadata_dir / f"{prompt_id}.json"
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "prompt_id": prompt_id,
                    "prompt_text": prompt_text,
                    "category": category,
                    "models": list(images_per_model.keys()),
                    "seed": args.seed,
                },
                f,
                indent=2,
            )

        res = EpistemicResult(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            category=category,
            models=list(images_per_model.keys()),
            mean_similarity=mean_sim,
            variance_similarity=var_sim,
        )

        results.append(asdict(res))

    # Save results as CSV
    df = pd.DataFrame(results)
    csv_path = out_dir / args.csv_name
    df.to_csv(csv_path, index=False)
    
    # Also save a summary by category
    if len(results) > 0:
        df_summary = df.groupby("category").agg({
            "mean_similarity": ["mean", "std"],
            "variance_similarity": ["mean", "std"],
        }).round(4)
        summary_path = out_dir / "summary_by_category.csv"
        df_summary.to_csv(summary_path)
        print(f"\nSaved category summary to {summary_path}")
    
    print(f"Saved results to {csv_path}")
    print(f"Total prompts processed: {len(results)}")


if __name__ == "__main__":
    main()

