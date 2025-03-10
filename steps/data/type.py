import os
from pathlib import Path
from typing import cast

from datasets import DatasetDict, Dataset, load_dataset, get_dataset_config_names

class EquiBenchDatasets:
    def __init__(self):
        self.path = "frogcjn/EquiBench-Datasets"

    @staticmethod
    def check_huggingface_login():
        """Check Hugging Face CLI login status"""
        print("Check Hugging Face CLI login status:")
        if os.system("huggingface-cli whoami") != 0:
            raise RuntimeError("Please log in to Hugging Face using 'huggingface-cli login'.")

    @property
    def config_names(self) -> list[str]:
        config_names = get_dataset_config_names(self.path)
        print(f"Dataset configurations: {config_names}")
        return config_names

    def load(self, config_name: str) -> Dataset:
        """Load and save a specific configuration of the EquiBench dataset"""
        dataset_dict = cast(DatasetDict, load_dataset("frogcjn/EquiBench-Datasets", name=config_name))
        print(f"Loading configuration: {config_name}")
        assert dataset_dict.keys() == {"train"}
        dataset = dataset_dict["train"]
        print(dataset)
        return dataset

    def save(self, config_name: str, dataset: Dataset, save_path: Path):
        """Save the loaded dataset to disk in JSON format with indent=4"""
        save_path.mkdir(parents=True, exist_ok=True)
        json_path = save_path / f"pairs.json"
        dataset.to_json(json_path, orient="records", lines=False, indent=4)
        print(f"Saved {config_name} to {json_path}")
