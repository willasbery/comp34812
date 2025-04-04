from pathlib import Path

class BaseConfig:
    DATA_DIR = Path(__file__).parent.parent / "data"
    TRAIN_FILE = DATA_DIR / "train.csv"
    DEV_FILE = DATA_DIR / "dev.csv"
    AUG_TRAIN_FILE = DATA_DIR / "train_augmented.csv"
    SAVE_DIR = DATA_DIR / "results"
    CACHE_DIR = Path(__file__).parent.parent / "cache"

    # Augmentation config
    AUGMENTATION_CONFIG = {
        "0": {
            "replace": 0.0,
            "add": 0.1,
            "translate":{
                "percentage": 0.8,
                "split": {
                    "Claim": 0.15,
                    "Evidence": 0.7,
                    "Both": 0.15
                },
                "src": "en",
                "intermediates": {
                    "fr": 0.5,
                    "de": 0.4,
                    "ja": 0.1
                }
            },
            "synonym_replacement": {
                "percentage": 0.8,
                "split": {
                    "Claim": 0.15,
                    "Evidence": 0.7,
                    "Both": 0.15
                }
            },
            "easy_data_augmentation": {
                "percentage": 0.8,
                "split": {
                    "Claim": 0.1,
                    "Evidence": 0.9,
                    "Both": 0.0
                }
            }
        },
        "1": {
            "replace": 0.0,
            "add": 2.0,
            "translate":{
                "percentage": 0.8,
                "split": {
                    "Claim": 0.15,
                    "Evidence": 0.7,
                    "Both": 0.15
                },
                "src": "en",
                "intermediates": {
                    "fr": 0.5,
                    "de": 0.4,
                    "ja": 0.1
                }
            },
            "synonym_replacement": {
                "percentage": 0.8,
                "split": {
                    "Claim": 0.15,
                    "Evidence": 0.7,
                    "Both": 0.15
                }
            },
            "easy_data_augmentation": {
                "percentage": 0.2,
                "split": {
                    "Claim": 0.1,
                    "Evidence": 0.9,
                    "Both": 0.0
                }
            }
        }
    }
    
def get_config() -> BaseConfig:
    return BaseConfig()


config = get_config()
    