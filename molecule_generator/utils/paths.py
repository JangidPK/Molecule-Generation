from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_SMILES = DATA_DIR / "smiles_list.txt"
CLEANED_SMILES = DATA_DIR / "cleaned_smiles.txt"
SELFIES_DATASET = DATA_DIR / "selfies_dataset.txt"

# Config
CONFIG_FILE = PROJECT_ROOT / "configs" / "config.yaml"

# Model directory (optional)
MODELS_DIR = PROJECT_ROOT / "models"

# saving the parameters
ARTF_DIR = PROJECT_ROOT / "artifacts" 