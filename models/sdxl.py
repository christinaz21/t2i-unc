# models/sdxl.py

import torch
from diffusers import StableDiffusionXLPipeline
from typing import List, Dict, Any
from PIL import Image


class SDXLModel:
    """
    Wrapper around Stable Diffusion XL using HuggingFace diffusers.
    Provides a unified interface so experiments can call:
        model.generate(prompt, num_images, seeds, return_latents=...)
    """

    def __init__(
        self,
        device: str = "cuda",
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
    ):
        self.name = "sdxl"
        self.device = device

        print(f"[SDXLModel] Loading {model_id}...")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            safety_checker=None,       
        )

        self.pipe = pipe.to(device)

        # Use the VAE's scaling_factor if available, fall back to SD default
        self.latent_scale = getattr(self.pipe.vae.config, "scaling_factor", 0.18215)

    def generate(
        self,
        prompt: str,
        num_images: int = 1,
        seeds: List[int] | None = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        return_latents: bool = False,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Generate `num_images` images for a prompt with optional fixed seeds.

        Returns a list of dicts:
        {
            "image": PIL.Image,
            "prompt": str,
            "seed": int,
            "metadata": {...},
            # "latents": torch.Tensor (C,H,W)   # only if return_latents=True
        }
        """

        results: List[Dict[str, Any]] = []

        # Generate seeds if not provided
        if seeds is None:
            seeds = [torch.randint(0, 2**32 - 1, ()).item() for _ in range(num_images)]

        assert len(seeds) == num_images, "num_images must match len(seeds)"

        for seed in seeds:
            generator = torch.Generator(device=self.device).manual_seed(seed)

            if not return_latents:
                # Normal SDXL pipeline -> returns PIL images
                out = self.pipe(
                    prompt=prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    **kwargs,
                )

                img: Image.Image = out.images[0]
                entry: Dict[str, Any] = {
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

            else:
                # Ask diffusers for the *latent* output
                out = self.pipe(
                    prompt=prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    output_type="latent",    # key: get final latents instead of decoded image
                    **kwargs,
                )

                latents = out.images[0]      # (C, H/8, W/8)

                # Decode latents manually using the VAE & scaling factor
                with torch.no_grad():
                    decoded = self.pipe.vae.decode(
                        latents.unsqueeze(0) / self.latent_scale
                    ).sample
                    decoded = (decoded / 2 + 0.5).clamp(0, 1)
                    img = Image.fromarray(
                        (decoded[0].permute(1, 2, 0).cpu().numpy() * 255).astype(
                            "uint8"
                        )
                    )

                entry = {
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
                    "latents": latents.detach().cpu(),
                }

            results.append(entry)

        return results
