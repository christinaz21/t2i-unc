from typing import List, Dict, Any
from models.base_t2i import BaseT2IModel
from src.utils.io import save_image_with_metadata

class ImageGenerator:
    def __init__(self, model: BaseT2IModel, out_dir: str):
        self.model = model
        self.out_dir = out_dir

    def generate_for_prompt(
        self,
        prompt: str,
        seeds: List[int],
        return_latents: bool = False,
        extra_meta: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        results = self.model.generate(prompt, num_images=len(seeds), seeds=seeds, return_latents=return_latents)
        for r in results:
            meta = {
                "model": self.model.name,
                "prompt": prompt,
                "seed": r["seed"],
            }
            if extra_meta:
                meta.update(extra_meta)
            save_image_with_metadata(
                image=r["image"],
                out_dir=self.out_dir,
                metadata=meta,
            )
        return results
