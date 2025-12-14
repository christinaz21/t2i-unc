# models/sdxl.py

import torch
from diffusers import StableDiffusionXLPipeline
from typing import List, Dict, Any
from PIL import Image
import torchvision.transforms as T
from diffusers import AutoencoderKL




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

        # Simple transform: PIL -> tensor in [-1, 1]
        self.to_tensor = T.ToTensor()

        self.vae_model = AutoencoderKL.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            subfolder = "vae",
            # torch_dtype=torch.float32 # for mps, the fp16 does not work
        )


    def _encode_image_to_latent(self, img: Image.Image) -> torch.Tensor:
        """
        Encode a PIL image into SDXL VAE latent space.
        Returns a tensor of shape (C, H/8, W/8) on CPU.
        """
        img = img.convert("RGB")
        # [0,1], shape (C,H,W)
        t = self.to_tensor(img).unsqueeze(0).to(self.device)  # (1,3,H,W)
        # Map to [-1,1] as VAE expects
        t = t * 2 - 1

        # 🔑 Make sure dtype matches VAE weights (likely float16)
        vae_dtype = next(self.pipe.vae.parameters()).dtype
        t = t.to(vae_dtype)

        with torch.no_grad():
            posterior = self.pipe.vae.encode(t)
            # latent_dist.mean has shape (1, C, H/8, W/8)
            latents = posterior.latent_dist.mean * self.latent_scale

        return latents[0].detach().cpu()  # (C,H/8,W/8)


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

                # # Ask diffusers for the *latent* output
                # out = self.pipe(
                #     prompt=prompt,
                #     guidance_scale=guidance_scale,
                #     num_inference_steps=num_inference_steps,
                #     generator=generator,
                #     output_type="latent",  # returns latents instead of PIL
                #     **kwargs,
                # )

                # latents = out.images[0]  # (C, H/8, W/8)

                # # Manually decode with VAE + correct scaling factor
                # with torch.no_grad():
                #     latents_batch = latents.unsqueeze(0)  # (1, C, H/8, W/8)
                #     decoded = self.pipe.vae.decode(
                #         latents_batch / self.latent_scale
                #     ).sample
                #     # decoded is in [-1,1] -> map to [0,1]
                #     decoded = (decoded / 2 + 0.5).clamp(0, 1)
                #     img_arr = (decoded[0].permute(1, 2, 0).cpu().numpy() * 255).astype(
                #         "uint8"
                #     )
                #     img = Image.fromarray(img_arr)

                # entry = {
                #     "image": img,
                #     "prompt": prompt,
                #     "seed": seed,
                #     "metadata": {
                #         "model": self.name,
                #         "prompt": prompt,
                #         "seed": seed,
                #         "steps": num_inference_steps,
                #         "guidance_scale": guidance_scale,
                #     },
                #     "latents": latents.detach().cpu(),
                # }


            if return_latents:
                # image_latent = self.vae_model.encode(img).latent_dist.sample()
                # print(image_latent.shape)
                # latents = self._encode_image_to_latent(img)
                # # entry["latents"] = latents
                # entry["latents"] = image_latent.detach().cpu()

                # This handles PIL / list of PILs etc and returns [B, C, H, W] in [0,1]
                img = self.pipe.image_processor.preprocess(img).to(self.device, dtype=next(self.pipe.vae.parameters()).dtype)
                img = img * 2.0 - 1.0  # [-1, 1]

                posterior = self.vae_model.encode(img)
                latents = posterior.latent_dist.sample()

                if hasattr(self.vae_model.config, "scaling_factor"):
                    latents = latents * self.vae_model.config.scaling_factor

                entry["latents"] = latents.detach().cpu()[0]

            results.append(entry)

        return results
