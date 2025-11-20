from dataclasses import dataclass
from typing import List

@dataclass
class PromptEntry:
    id: str
    text: str
    category: str      # "specific", "moderate", "abstract", "metaphor"

class PromptDataset:
    def __init__(self, jsonl_path: str):
        self.entries: List[PromptEntry] = self._load(jsonl_path)

    def filter_by_category(self, category: str) -> List[PromptEntry]:
        return [p for p in self.entries if p.category == category]

    def all_categories(self) -> List[str]:
        return sorted({p.category for p in self.entries})
