# models/pixart_alpha.py

import torch
from diffusers import PixArtAlphaPipeline
from transformers import T5Tokenizer, T5EncoderModel
from typing import List, Dict, Any
from PIL import Image


class PixArtModel:
    """
    Wrapper around PixArt Alpha using HuggingFace diffusers.
    Provides a unified interface so experiments can call:
        model.generate(prompt, num_images, seeds, return_latents=...)
    """

    def __init__(
        self,
        device: str = "cuda",
        model_id: str = "PixArt-alpha/PixArt-XL-2-512x512",
    ):
        self.name = "pixart"
        self.device = device

        print(f"[PixArtModel] Loading {model_id}...")
        
        # PixArt Alpha requires T5 tokenizer/encoder
        # Check if sentencepiece is available for explicit loading
        try:
            import sentencepiece
            # Load T5 components explicitly if sentencepiece is available
            t5_model_name = "google/t5-v1_1-xxl"
            print(f"[PixArtModel] Loading T5 tokenizer and encoder ({t5_model_name})...")
            tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
            # Use safetensors format to avoid torch version requirement
            text_encoder = T5EncoderModel.from_pretrained(
                t5_model_name,
                torch_dtype=torch.float16 if "cuda" in device else torch.float32,
                use_safetensors=True,
            ).to(device)
            
            # Load pipeline with explicit components
            pipe = PixArtAlphaPipeline.from_pretrained(
                model_id,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            )
        except ImportError:
            # SentencePiece not available - try loading tokenizer from model repo
            print(f"[PixArtModel] SentencePiece not available. Trying to load from model repository...")
            try:
                # Try loading tokenizer from the model's own repository if it exists
                tokenizer = T5Tokenizer.from_pretrained(model_id, subfolder="tokenizer", use_fast=False)
                text_encoder = T5EncoderModel.from_pretrained(
                    model_id,
                    subfolder="text_encoder",
                    torch_dtype=torch.float16 if "cuda" in device else torch.float32,
                    use_safetensors=True,
                ).to(device)
                
                pipe = PixArtAlphaPipeline.from_pretrained(
                    model_id,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    torch_dtype=torch.float16 if "cuda" in device else torch.float32,
                )
            except Exception as e2:
                print(f"[PixArtModel] Could not load tokenizer from model repo: {e2}")
                print(f"[PixArtModel] Note: PixArt Alpha requires sentencepiece library.")
                print(f"[PixArtModel] Please install it with: pip install sentencepiece")
                raise ImportError(
                    "PixArt Alpha requires sentencepiece library. "
                    "Please install it with: pip install sentencepiece"
                ) from e2
        except Exception as e:
            print(f"[PixArtModel] Error loading T5 components: {e}")
            raise

        self.pipe = pipe.to(device)

    def generate(
        self,
        prompt: str,
        num_images: int = 1,
        seeds: List[int] | None = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,
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
            "metadata": {...},
            "latents": torch.Tensor (optional)
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
                # Generate image directly
                out = self.pipe(
                    prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    **kwargs,
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
                # Generate with latents
                out = self.pipe(
                    prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    output_type="latent",
                    **kwargs,
                )

                latents = out.images[0]

                # Decode latents to image
                with torch.no_grad():
                    # PixArt uses a scaling factor similar to SD
                    scaling_factor = getattr(self.pipe.vae.config, "scaling_factor", 0.18215)
                    decoded = self.pipe.vae.decode(latents.unsqueeze(0) / scaling_factor).sample
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

