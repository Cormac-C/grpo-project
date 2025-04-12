from torch.utils.data import Dataset
import json
from typing import List, Dict
from datasets import load_dataset

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

        preamble = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
        User: Using the numbers {}, create an equation that equals {}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
        Assistant: Let me solve this step by step.
        <think>"""

        sample = {
            # TODO: Maybe look at changing this prompt to be more like tinyzero
            "prompt": preamble.format(numbers, target),
            "numbers": numbers,
            "target": target,
        }

        return sample
    

class Countdown_HF(Dataset):
    def __init__(self):
        """
        PyTorch dataset for the Countdown problem.
        Loads data from a dict. Each entry must have 'target' and 'nums'.
        """
        DATASET_NAME = "Jiayi-Pan/Countdown-Tasks-3to4"
        self.data: List[Dict] = load_dataset(DATASET_NAME, split="train")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        target = item["target"]
        numbers = item["nums"]

        preamble = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
        User: Using the numbers {}, create an equation that equals {}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
        Assistant: Let me solve this step by step.
        <think>"""

        sample = {
            # TODO: Maybe look at changing this prompt to be more like tinyzero
            "prompt": preamble.format(numbers, target),
            "numbers": numbers,
            "target": target,
        }

        return sample
