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
            safety_checker=None 
        )

        self.pipe = pipe.to(device)

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

            if not return_latents:

                out = self.pipe(
                    prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                )

                img: Image.Image = out.images[0]

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
                    }
                
            
            else:

                out = self.pipe(
                    prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    output_type="latent",
                )

                latents = out.images[0]

                # NOTE: SD uses VAE decode on latents scaled by 1/0.18215
                with torch.no_grad():
                    decoded = self.pipe.vae.decode(latents.unsqueeze(0) / 0.18215).sample
                    decoded = (decoded / 2 + 0.5).clamp(0, 1)
                    img = Image.fromarray(
                        (decoded[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
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
                }

                if return_latents:
                    entry["latents"] = latents.detach().cpu()

            results.append(entry)
        return results
