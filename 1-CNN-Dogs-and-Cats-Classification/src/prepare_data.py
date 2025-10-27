"""
prepare_data.py
Downloading a dataset using the Kaggle API with the personal kaggle.json file.
Usage:
    python src/prepare_data.py
"""

import os
import subprocess

def download_dataset(dataset="tongpython/cat-and-dog", output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    # Kaggle API key verification
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        raise FileNotFoundError(
            "⚠️ kaggle.json could not be found. Please add it to the ~/.kaggle/ directory."
        )
    print("⏳ Data downloading...")
    subprocess.run(f"kaggle datasets download -d {dataset} -p {output_dir} --unzip", shell=True, check=True)
    print(f"✅ Data downloaded and {output_dir} directory has been opened.")

if __name__ == "__main__":
    download_dataset()

