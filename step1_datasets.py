# huggingface-cli login
# huggingface-cli whoami

from pathlib import Path
from steps.datasets import EquiBenchDatasets

def main():
    datasets = EquiBenchDatasets()

    # Check Hugging Face CLI login status
    datasets.check_huggingface_login()
    
    # Load dataset configuration names
    config_names = datasets.config_names
    
    # Load and save each dataset configuration
    datasets_path = Path("data")
    for config_name in config_names:
        dataset_dict = datasets.load(config_name=config_name)
        datasets.save(dataset_dict=dataset_dict, config_name=config_name, save_path=datasets_path / config_name)

if __name__ == "__main__":
    main()
