from dataclasses import dataclass
from typing import List
import json

@dataclass
class PromptEntry:
    id: str
    text: str
    category: str      # "specific", "moderate", "abstract", "metaphor"

class PromptDataset:
    def __init__(self, jsonl_path: str):
        self.entries: List[PromptEntry] = self._load(jsonl_path)

    def _load(self, jsonl_path: str) -> List[PromptEntry]:
        entries: List[PromptEntry] = []
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                # Expecting keys: "id", "text", "category"
                entries.append(
                    PromptEntry(
                        id=obj["id"],
                        text=obj["text"],
                        category=obj["category"],
                    )
                )
        return entries

    def filter_by_category(self, category: str) -> List[PromptEntry]:
        return [p for p in self.entries if p.category == category]

    def all_categories(self) -> List[str]:
        return sorted({p.category for p in self.entries})
    
    def print_prompts(self):
        for p in self.entries:
            print(f"{p.id} ({p.category}): {p.text}")
