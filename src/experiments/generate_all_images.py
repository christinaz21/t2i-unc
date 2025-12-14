#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from src.prompts.prompt_dataset import PromptDataset
from src.generation.generator import ImageGenerator
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
        description="Generate images for all prompts across multiple prompt files"
    )
    
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default="/u/cz5047/reas-llm-uq/t2i-unc/prompts",
        help="Directory containing prompt JSONL files.",
    )
    
    parser.add_argument(
        "--prompt_files",
        type=str,
        nargs="+",
        default=["abstract.jsonl", "concrete.jsonl", "moderate.jsonl", "underspecified.jsonl"],
        help="List of prompt files to process.",
    )
    
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/generated_images",
        help="Base directory to store generated images.",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="sdxl",
        choices=list(MODEL_REGISTRY.keys()),
        help="Which T2I model to use (sd15, sdxl, or pixart).",
    )
    
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=8,
        help="Number of seeds (images) to generate per prompt.",
    )
    
    parser.add_argument(
        "--seed_offset",
        type=int,
        default=0,
        help="Offset for random seeds (seed = seed_offset + i).",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for T2I model (e.g., 'cuda' or 'cpu').",
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable image caching (regenerate all images).",
    )
    
    parser.add_argument(
        "--save_progress",
        type=str,
        default=None,
        help="Optional path to save progress CSV (e.g., 'progress.csv').",
    )
    
    return parser.parse_args()


def make_seeds(num_seeds: int, offset: int) -> List[int]:
    """Generate list of seeds."""
    return [offset + i for i in range(num_seeds)]


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def main():
    args = parse_args()
    
    # Load all prompts from specified files
    prompt_dir = Path(args.prompt_dir)
    all_prompts = []
    
    print(f"Loading prompts from {len(args.prompt_files)} files...")
    for prompt_file in args.prompt_files:
        prompt_path = prompt_dir / prompt_file
        if not prompt_path.exists():
            print(f"Warning: Prompt file {prompt_path} does not exist, skipping.")
            continue
        
        dataset = PromptDataset(str(prompt_path))
        all_prompts.extend(dataset.entries)
        print(f"  Loaded {len(dataset.entries)} prompts from {prompt_file}")
    
    if len(all_prompts) == 0:
        raise ValueError("No prompts loaded. Check --prompt_dir and --prompt_files.")
    
    print(f"\nTotal prompts to process: {len(all_prompts)}")
    print(f"Images per prompt: {args.num_seeds}")
    print(f"Total images to generate: {len(all_prompts) * args.num_seeds}")
    
    # Set up output directory
    out_dir = ensure_dir(args.out_dir)
    
    # Instantiate model first to get model name
    ModelCls = MODEL_REGISTRY[args.model]
    t2i_model = ModelCls(device=args.device)
    
    # Create model-specific image directory
    images_dir = ensure_dir(out_dir / "images" / t2i_model.name)
    
    # Save configuration
    config_path = out_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"\nConfiguration saved to {config_path}")
    
    # Create generator with model-specific directory
    use_cache = not args.no_cache
    generator = ImageGenerator(
        model=t2i_model,
        out_dir=str(images_dir),
        use_cache=use_cache
    )
    
    if use_cache:
        print("Caching enabled: existing images will be reused.")
    else:
        print("Caching disabled: all images will be regenerated.")
    
    # Progress tracking
    progress_data = []
    seeds = make_seeds(args.num_seeds, args.seed_offset)
    
    print(f"\nGenerating images with seeds: {seeds[:5]}{'...' if len(seeds) > 5 else ''}")
    print("=" * 80)
    
    # Process all prompts
    for prompt_entry in tqdm(all_prompts, desc="Prompts"):
        prompt_id = prompt_entry.id
        prompt_text = prompt_entry.text
        category = prompt_entry.category
        
        try:
            # Generate images (will use cache if available)
            gen_results = generator.generate_for_prompt(
                prompt_text,
                seeds=seeds,
                prompt_id=prompt_id,
                extra_meta={
                    "prompt_id": prompt_id,
                    "category": category,
                    "batch_generation": True,
                },
                return_latents=False,
            )
            
            # Track progress
            num_generated = len(gen_results)
            progress_data.append({
                "prompt_id": prompt_id,
                "prompt_text": prompt_text,
                "category": category,
                "model": t2i_model.name,
                "num_seeds": args.num_seeds,
                "num_generated": num_generated,
                "status": "success" if num_generated == args.num_seeds else "partial",
            })
            
        except Exception as e:
            print(f"\nError processing prompt {prompt_id}: {e}")
            progress_data.append({
                "prompt_id": prompt_id,
                "prompt_text": prompt_text,
                "category": category,
                "model": t2i_model.name,
                "num_seeds": args.num_seeds,
                "num_generated": 0,
                "status": f"error: {str(e)[:100]}",
            })
    
    # Save progress report
    if args.save_progress:
        progress_path = Path(args.save_progress)
    else:
        progress_path = out_dir / "generation_progress.csv"
    
    df = pd.DataFrame(progress_data)
    df.to_csv(progress_path, index=False)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Generation Summary:")
    print(f"  Total prompts processed: {len(all_prompts)}")
    print(f"  Successful: {len([p for p in progress_data if p['status'] == 'success'])}")
    print(f"  Partial: {len([p for p in progress_data if p['status'] == 'partial'])}")
    print(f"  Errors: {len([p for p in progress_data if 'error' in p['status']])}")
    print(f"  Total images generated: {sum(p['num_generated'] for p in progress_data)}")
    print(f"\nProgress saved to: {progress_path}")
    print(f"Images saved to: {images_dir} (model: {t2i_model.name})")


if __name__ == "__main__":
    main()

