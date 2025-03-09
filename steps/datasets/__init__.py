import os
from pathlib import Path
from typing import cast

from datasets import DatasetDict, load_dataset, get_dataset_config_names

class EquiBenchDatasets:
    def __init__(self):
        self.path = "frogcjn/EquiBench-Datasets"

    @staticmethod
    def check_huggingface_login():
        """Check Hugging Face CLI login status"""
        if os.system("huggingface-cli whoami") != 0:
            raise RuntimeError("Please log in to Hugging Face using 'huggingface-cli login'.")

    @property
    def config_names(self) -> list[str]:
        config_names = get_dataset_config_names(self.path)
        print(f"Dataset configurations: {config_names}")
        return config_names

    def load(self, config_name: str) -> DatasetDict:
        """Load and save a specific configuration of the EquiBench dataset"""
        dataset = cast(DatasetDict, load_dataset("frogcjn/EquiBench-Datasets", name=config_name))
        print(f"Loading configuration: {config_name}")
        print(dataset)
        return dataset

    def save(self, config_name: str, dataset_dict: DatasetDict, save_path: Path):
        """Save the loaded dataset to disk"""
        dataset_dict.save_to_disk(save_path)
        print(f"Saving dataset to {save_path}")
        print(f"Saved dataset configuration '{config_name}' to {save_path}")

__all__ = ["EquiBenchDatasets"]
