from pathlib import Path

class BaseConfig:
    DATA_DIR = Path(__file__).parent.parent / "data"
    TRAIN_FILE = DATA_DIR / "train.csv"
    DEV_FILE = DATA_DIR / "dev.csv"
    TEST_FILE = DATA_DIR / "test.csv"
    AUG_TRAIN_FILE = DATA_DIR / "train_augmented.csv"
    SAVE_DIR = DATA_DIR / "results"
    CACHE_DIR = Path(__file__).parent.parent / "cache"

    # Augmentation config
    AUGMENTATION_CONFIG = {
        "0": {
            "replace": 0.0,
            "add": 0.1, # 10%
            "translate":{
                "percentage": 1.0,
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
                "percentage": 0.7,
                "replacement_fraction": 0.3,
                "min_similarity": 0.85,
                "min_word_length": 4,
                "word_frequency_threshold": 3,
                "synonym_selection_strategy": "random",
                "enable_random_synonym_insertion": True,
                "synonym_insertion_probability": 0.03,
                "enable_random_word_insertion": True,
                "word_insertion_probability": 0.01,
                "enable_random_deletion": True,
                "deletion_probability": 0.01,
            },
            "x_or_y": {
                "percentage": 0.08,
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
            "add": 1.0,
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
                "percentage": 0.7,
                "replacement_fraction": 0.3,
                "min_similarity": 0.85,
                "min_word_length": 4,
                "word_frequency_threshold": 3,
                "synonym_selection_strategy": "random",
                "enable_random_synonym_insertion": True,
                "synonym_insertion_probability": 0.03,
                "enable_random_word_insertion": True,
                "word_insertion_probability": 0.01,
                "enable_random_deletion": True,
                "deletion_probability": 0.01,
            },
            "x_or_y": {
                "percentage": 0.02,
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
        }
    }
    
def get_config() -> BaseConfig:
    return BaseConfig()


config = get_config()
    