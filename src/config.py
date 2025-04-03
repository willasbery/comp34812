from pathlib import Path

class BaseConfig:
    DATA_DIR = Path(__file__).parent.parent / "data"
    TRAIN_FILE = DATA_DIR / "train.csv"
    DEV_FILE = DATA_DIR / "dev.csv"
    AUG_TRAIN_FILE = DATA_DIR / "train_augmented.csv"
    AUG_TRAIN_HIGH_REPLACEMENT_FILE = DATA_DIR / "train_augmented_high_replacement_fraction.csv"
    SAVE_DIR = DATA_DIR / "results"
    CACHE_DIR = Path(__file__).parent.parent / "cache"
    
def get_config() -> BaseConfig:
    return BaseConfig()


config = get_config()
    