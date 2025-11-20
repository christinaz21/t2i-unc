import torch
from diffusers import StableDiffusionPipeline
from typing import List, Dict, Any
from PIL import Image


class StableDiffusionModel:
    """
    Wrapper around Stable Diffusion v1.5 using HuggingFace diffusers.
    Provides a unified interface so experiments can call:
        model.generate(prompt, num_images, seeds)
    """

    def __init__(self, device: str = "cuda", model_id: str = "runwayml/stable-diffusion-v1-5"):
        self.name = "sd15"
        self.device = device

        # Load pretrained pipeline
        print(f"[StableDiffusionModel] Loading {model_id}...")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            safety_checker=None  # Disable NSFW filter for research settings
        )

        self.pipe = pipe.to(device)

    def generate(
        self,
        prompt: str,
        num_images: int = 1,
        seeds: List[int] | None = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Generate `num_images` images for a prompt with optional fixed seeds.

        Returns list of dicts:
        {
            "image": PIL.Image,
            "prompt": str,
            "seed": int,
            "metadata": {...}
        }
        """

        results = []

        if seeds is None:
            # Auto-generate unique seeds
            seeds = [torch.randint(0, 2**32 - 1, ()).item() for _ in range(num_images)]

        assert len(seeds) == num_images, "num_images must match len(seeds)"

        for seed in seeds:
            generator = torch.Generator(device=self.device).manual_seed(seed)

            out = self.pipe(
                prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            )

            img: Image.Image = out.images[0]

            results.append(
                {
                    "image": img,
                    "prompt": prompt,
                    "seed": seed,
                    "metadata": {
                        "model": self.name,
                        "prompt": prompt,
                        "seed": seed,
                        "steps": num_inference_steps,
                        "guidance_scale": guidance_scale,
                    },
                }
            )

        return results
