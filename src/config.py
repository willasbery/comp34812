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
            "add": 0.001,
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
                "replacement_fraction": 0.3,
                "min_similarity": 0.85,
                "min_word_length": 4,
                "word_frequency_threshold": 1,
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
            },
            "x_or_y": {
                "percentage": 0.3,
                "max_choices": 4,
                "num_words_to_augment": {
                    "Claim": 1,
                    "Evidence": 2
                },
                "split": {
                    "Claim": 0.90,
                    "Evidence": 0.05,
                    "Both": 0.05
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
                "replacement_fraction": 0.3,
                "min_similarity": 0.85,
                "min_word_length": 4,
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
            },
            "x_or_y": {
                "percentage": 0.5,
                "max_choices": 4,
                "num_words_to_augment": {
                    "Claim": 1,
                    "Evidence": 1
                },
                "split": {
                    "Claim": 0.90,
                    "Evidence": 0.05,
                    "Both": 0.05
                }
            }
        }
    }
    
def get_config() -> BaseConfig:
    return BaseConfig()


config = get_config()
    