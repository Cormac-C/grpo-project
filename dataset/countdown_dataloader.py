from torch.utils.data import Dataset
import json
from typing import List, Dict


class Countdown(Dataset):
    def __init__(self, json_path: str):
        """
        PyTorch dataset for the Countdown problem.
        Loads data from a JSON file. Each entry must have 'target' and 'numbers'.
        """
        with open(json_path, "r") as f:
            self.data: List[Dict] = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        target = item["target"]
        numbers = item["numbers"]

        sample = {
            # TODO: Maybe look at changing this prompt to be more like tinyzero
            "prompt": f"Using the numbers {item["numbers"]}, create an equation that equals {item["target"]}. Box your answer.",
            "numbers": numbers,
            "target": target,
        }

        return sample
