# models/lpips_scorer.py

from __future__ import annotations
from typing import Union

import torch
import lpips
from PIL import Image
import torchvision.transforms as T


class LPIPSScorer:
    """
    Thin wrapper around the lpips library.

    Uses a learned perceptual network (AlexNet by default) to compute
    LPIPS distance between images. Higher distance = more perceptually different.
    """

    def __init__(self, net: str = "alex", device: str = "cuda"):
        """
        Args:
            net: backbone used by LPIPS ("alex", "vgg", "squeeze")
            device: "cuda" or "cpu"
        """
        self.device = device
        self.lpips = lpips.LPIPS(net=net).to(device)
        self.lpips.eval()

        # Standard transform: PIL -> [0,1] tensor -> [-1,1] normalized
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    def _to_lpips_tensor(self, img: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """
        Convert PIL image or tensor to a 4D tensor (1,3,H,W) in [-1, 1]
        on the correct device, as expected by lpips.
        """
        # Case 1: PIL image
        if isinstance(img, Image.Image):
            img = img.convert("RGB")
            t = self.to_tensor(img)  # (3,H,W), in [0,1]
            t = self.normalize(t)    # -> [-1,1]
            t = t.unsqueeze(0)       # (1,3,H,W)
            return t.to(self.device)

        # Case 2: torch.Tensor
        if isinstance(img, torch.Tensor):
            t = img
            # If HWC, convert to CHW
            if t.ndim == 3 and t.shape[0] not in (1, 3):
                # Assume HWC
                t = t.permute(2, 0, 1)  # (C,H,W)

            # Take the first in batch since LPIPS expects (1,3,H,W)
            if t.ndim == 4:
                t = t[0]

            # Scale to [0,1] if it looks like [0,255]
            if t.max() > 1.5:
                t = t / 255.0

            # Now normalize to [-1,1]
            if t.ndim == 3:
                t = self.normalize(t)
                t = t.unsqueeze(0)  # (1,3,H,W)
            elif t.ndim == 4:
                # already batched; normalize channel-wise
                t_list = []
                for i in range(t.shape[0]):
                    t_list.append(self.normalize(t[i]))
                t = torch.stack(t_list, dim=0)
            else:
                raise ValueError(f"Unexpected tensor shape for image: {t.shape}")

            return t.to(self.device)

        raise TypeError(f"Unsupported image type for LPIPS: {type(img)}")

    @torch.no_grad()
    def distance(self, img1, img2) -> float:
        """
        LPIPS distance between img1 and img2.
        Higher = more perceptually different.
        """
        t1 = self._to_lpips_tensor(img1)
        t2 = self._to_lpips_tensor(img2)
        d = self.lpips(t1, t2)
        return float(d.item())
