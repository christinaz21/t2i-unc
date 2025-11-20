# models/clip_encoder.py

import torch
from PIL import Image
from typing import Union
from transformers import CLIPProcessor, CLIPModel


class ClipScorer:
    """
    Wrapper around CLIP for computing:
        - image-text similarity
        - image-image similarity

    Similarities returned as scalar floats (cosine similarity).
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str = "cuda",
    ):
        self.device = device

        # Load CLIP model + preprocessor
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Put in eval mode
        self.model.eval()

    def _ensure_pil(self, img: Union[Image.Image, torch.Tensor]) -> Image.Image:
        if isinstance(img, Image.Image):
            return img
        elif isinstance(img, torch.Tensor):
            # Convert CHW or HWC to PIL Image
            if img.ndim == 3:
                # Clip values and move to CPU
                img = img.detach().cpu()
                if img.shape[0] == 3:  # CHW
                    img = img.permute(1, 2, 0)
                img = (img * 255).clamp(0, 255).byte().numpy()
                return Image.fromarray(img)
        raise TypeError(f"Unsupported image type for CLIP scoring: {type(img)}")

    # Image ↔ Text Similarity
    @torch.no_grad()
    def image_text_similarity(self, image, text: str) -> float:
        """
        Return cosine similarity between image and text.
        """
        image = self._ensure_pil(image)

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        output = self.model(**inputs)

        # CLIP returns pooled embeddings:
        img_emb = output.image_embeds  # shape: [1, D]
        txt_emb = output.text_embeds   # shape: [1, D]

        # Normalize for cosine similarity
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

        sim = (img_emb * txt_emb).sum().item()
        return float(sim)

    # Image ↔ Image Similarity
    @torch.no_grad()
    def image_image_similarity(self, img1, img2) -> float:
        """
        Compute cosine similarity between two images.
        """
        img1 = self._ensure_pil(img1)
        img2 = self._ensure_pil(img2)

        inputs = self.processor(
            images=[img1, img2],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        output = self.model.get_image_features(**inputs)

        # output has shape [2, D]
        img1_emb = output[0]
        img2_emb = output[1]

        # Normalize for cosine similarity
        img1_emb = img1_emb / img1_emb.norm()
        img2_emb = img2_emb / img2_emb.norm()

        sim = (img1_emb * img2_emb).sum().item()
        return float(sim)



    @torch.no_grad()
    def text_text_similarity(self, text1: str, text2: str) -> float:
        """
        Cosine similarity between two text strings in CLIP text embedding space.
        """
        inputs = self.processor(
            text=[text1, text2],
            images=None,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Get text features only
        text_embeds = self.model.get_text_features(**{k: inputs[k] for k in ["input_ids", "attention_mask"]})
        # text_embeds: (2, D)
        t1 = text_embeds[0]
        t2 = text_embeds[1]

        # Normalize and cosine similarity
        t1 = t1 / t1.norm()
        t2 = t2 / t2.norm()
        sim = (t1 * t2).sum().item()
        return float(sim)
