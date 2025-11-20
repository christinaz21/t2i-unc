from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseT2IModel(ABC):
    name: str

    @abstractmethod
    def generate(
        self,
        prompt: str,
        num_images: int = 1,
        seeds: List[int] | None = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Returns a list of dicts with:
        {
            "image": PIL.Image,
            "prompt": str,
            "seed": int,
            "metadata": {...}
        }
        """
        ...
