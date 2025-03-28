from pathlib import Path

class BaseConfig:
    DATA_DIR = Path(__file__).parent.parent / "data"
    TRAIN_FILE = DATA_DIR / "train.csv"
    DEV_FILE = DATA_DIR / "dev.csv"
    AUG_TRAIN_FILE = DATA_DIR / "train_augmented.csv"
    SAVE_DIR = DATA_DIR / "results"

    
def get_config() -> BaseConfig:
    return BaseConfig()


config = get_config()
    