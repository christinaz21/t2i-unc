from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from models.base_t2i import BaseT2IModel
from src.utils.io import (
    save_image_with_metadata,
    get_image_filename,
    get_latent_filename,
    load_cached_image,
    load_cached_latent,
    save_latent,
)

class ImageGenerator:
    def __init__(self, model: BaseT2IModel, out_dir: str, use_cache: bool = True):
        self.model = model
        self.out_dir = Path(out_dir)
        self.use_cache = use_cache
        # Create latents directory if needed
        if use_cache:
            (self.out_dir / "latents").mkdir(parents=True, exist_ok=True)

    def generate_for_prompt(
        self,
        prompt: str,
        seeds: List[int],
        prompt_id: Optional[str] = None,
        return_latents: bool = False,
        extra_meta: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate images for a prompt, using cached images if available.
        
        Args:
            prompt: Text prompt
            seeds: List of seeds to use
            prompt_id: Optional prompt ID for caching (if None, will try to extract from extra_meta)
            return_latents: Whether to return latent tensors
            extra_meta: Additional metadata (should contain prompt_id if using caching)
            
        Returns:
            List of dicts with "image", "seed", "prompt", and optionally "latents"
        """
        # Extract prompt_id from extra_meta if not provided
        if prompt_id is None and extra_meta:
            prompt_id = extra_meta.get("prompt_id")
        
        results = []
        seeds_to_generate = []
        
        # Check which images are already cached
        for seed in seeds:
            if self.use_cache and prompt_id:
                image_path = get_image_filename(prompt_id, seed, self.model.name, self.out_dir)
                cached_image = load_cached_image(image_path)
                
                if cached_image is not None:
                    # Image exists, load it
                    result = {
                        "image": cached_image,
                        "prompt": prompt,
                        "seed": seed,
                        "metadata": {
                            "model": self.model.name,
                            "prompt": prompt,
                            "seed": seed,
                        },
                    }
                    
                    # Try to load cached latent if needed
                    if return_latents:
                        latent_path = get_latent_filename(prompt_id, seed, self.model.name, self.out_dir / "latents")
                        cached_latent = load_cached_latent(latent_path)
                        if cached_latent is not None:
                            result["latents"] = cached_latent
                        else:
                            # Latent not cached, we'll need to generate for this seed
                            seeds_to_generate.append(seed)
                            continue
                    
                    if extra_meta:
                        result["metadata"].update(extra_meta)
                    
                    results.append(result)
                    continue
            
            # Image not cached or caching disabled, need to generate
            seeds_to_generate.append(seed)
        
        print("Seeds to generate:", seeds_to_generate)
        
        # Generate images for seeds that aren't cached
        if seeds_to_generate:
            gen_results = self.model.generate(
                prompt, 
                num_images=len(seeds_to_generate), 
                seeds=seeds_to_generate, 
                return_latents=return_latents
            )
            
            for r in gen_results:
                seed = r["seed"]
                meta = {
                    "model": self.model.name,
                    "prompt": prompt,
                    "seed": seed,
                }
                if extra_meta:
                    meta.update(extra_meta)
                
                # If using cache and we have prompt_id, save with deterministic name for caching
                if self.use_cache and prompt_id:
                    image_path = get_image_filename(prompt_id, seed, self.model.name, self.out_dir)
                    r["image"].save(image_path)
                    
                    # Also save metadata JSON next to the cached image
                    meta_path = image_path.with_suffix(".json")
                    with open(meta_path, "w") as f:
                        json.dump(meta, f, indent=2)
                    
                    # Save latent if available
                    if return_latents and "latents" in r:
                        latent_path = get_latent_filename(prompt_id, seed, self.model.name, self.out_dir / "latents")
                        save_latent(r["latents"], latent_path)
                else:
                    # If not using cache, use the original save function
                    save_image_with_metadata(
                        image=r["image"],
                        out_dir=self.out_dir,
                        metadata=meta,
                    )
                
                results.append(r)
        
        # Sort results by seed to maintain consistent ordering
        results.sort(key=lambda x: x["seed"])
        return results
