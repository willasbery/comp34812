"""
PLAN:
1. specify the percentage of data to augment for each label and wether to replace or add
2. Load training data
3. basic cleaning and preprocessing (this will be done to test and validation data as well)
4. back translation
5. synonym replacement
6. synonym addition (trying to replicate the use of "/" eg: x leads to y/x)
7. random swap, deletion, insertion for 0 label types
8. save augmented data to a new file with orignal training data as well
"""
import logging
import random
import pandas as pd
import numpy as np
from src.config import config
from src.augmentation.back_translation.main import back_translate_batch
from src.augmentation.synonym_replacement.all_MiniLM_l6_v2.v2.main import AdvancedSynonymReplacerDF

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

def generate_augmented_samples(df: pd.DataFrame, label_counts: np.int64, num_samples: int) -> list[pd.DataFrame.index]:
    indices = []
    if num_samples > label_counts:
        full_repeats = num_samples // label_counts
        indices.extend(df.index.repeat(full_repeats))
        num_samples %= label_counts
    if num_samples > 0:
        indices.extend(df.sample(num_samples).index)
    return indices

async def back_translate_samples(aug_df: pd.DataFrame, label: str) -> pd.DataFrame:
    src = config.AUGMENTATION_CONFIG[label]["translate"]["src"]

    percentage_to_translate = config.AUGMENTATION_CONFIG[label]["translate"]["percentage"]
    samples = aug_df.sample(frac=percentage_to_translate)

    splits = config.AUGMENTATION_CONFIG[label]["translate"]["split"]
    languages = config.AUGMENTATION_CONFIG[label]["translate"]["intermediates"]

    claim_count = int(len(samples) * splits["Claim"])
    evidence_count = int(len(samples) * splits["Evidence"])

    split_samples = {
        "Claim": samples.iloc[:claim_count],
        "Evidence": samples.iloc[claim_count:claim_count+evidence_count],
        "Both": samples.iloc[claim_count+evidence_count:]
    }

    for text_type, sample in split_samples.items():
        count = 0
        for lang, percentage in languages.items():
            if count + int(len(sample) * percentage) > len(sample):
                aug_df.update(await back_translate_batch(sample.iloc[count:], text_type, src, lang))
                break
            aug_df.update(await back_translate_batch(sample.iloc[count:count + int(len(sample) * percentage)], text_type, src, lang))
            count += int(len(sample) * percentage)

    return aug_df
    
def synonym_replace_samples(aug_df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Apply synonym replacement augmentation to the specified samples.
    Modifies the input DataFrame in-place.
    
    Args:
        aug_df (pd.DataFrame): DataFrame containing samples to augment
        label (str): Label identifier ("0" or "1") to get config parameters
        
    Returns:
        pd.DataFrame: The modified input DataFrame
    """
    logging.info(f"Starting synonym replacement for label {label}")
    
    params = {
        "replacement_fraction": config.AUGMENTATION_CONFIG[label]["synonym_replacement"]["replacement_fraction"],
        "min_sentence_similarity": config.AUGMENTATION_CONFIG[label]["synonym_replacement"]["min_similarity"],
        "min_word_length": config.AUGMENTATION_CONFIG[label]["synonym_replacement"]["min_word_length"],
        "synonym_selection_strategy": "random",
        "word_frequency_threshold": 3,
        "enable_random_insertion": False,
        "enable_random_deletion": False
    }
    percentage_to_translate = config.AUGMENTATION_CONFIG[label]["synonym_replacement"]["percentage"]
    samples = aug_df.sample(frac=percentage_to_translate)
    
    replacer = AdvancedSynonymReplacerDF(params, samples)
    replacer.augment_data()  # This now modifies aug_df directly
    
    logging.info(f"Completed synonym replacement for label {label}")
    aug_df.update(samples)
    return aug_df  # Return the modified DataFrame

async def main():
    aug_df = pd.read_csv(config.TRAIN_FILE)
    aug_path = config.AUG_TRAIN_FILE

    # get label counts
    label_counts = aug_df['label'].value_counts()
    logging.info(f"Label counts: {label_counts}")

    zeros_to_replace = int(label_counts[0] * config.AUGMENTATION_CONFIG["0"]["replace"])
    ones_to_replace = int(label_counts[1] * config.AUGMENTATION_CONFIG["1"]["replace"])

    zeros_to_add = int(label_counts[0] * config.AUGMENTATION_CONFIG["0"]["add"])
    ones_to_add = int(label_counts[1] * config.AUGMENTATION_CONFIG["1"]["add"])

    logging.info(f"Zeros to replace: {zeros_to_replace}")
    logging.info(f"Ones to replace: {ones_to_replace}")
    logging.info(f"Zeros to add: {zeros_to_add}")
    logging.info(f"Ones to add: {ones_to_add}")

    # get the indices of the zeros and ones to replace
    zeros_to_replace_indices = generate_augmented_samples(aug_df[aug_df['label'] == 0], label_counts[0], zeros_to_replace)
    ones_to_replace_indices = generate_augmented_samples(aug_df[aug_df['label'] == 1], label_counts[1], ones_to_replace)
    zeros_to_add_indices = generate_augmented_samples(aug_df[aug_df['label'] == 0], label_counts[0], zeros_to_add)
    ones_to_add_indices = generate_augmented_samples(aug_df[aug_df['label'] == 1], label_counts[1], ones_to_add)

    #generate addition df
    ones_to_add_df = aug_df.iloc[ones_to_add_indices].reset_index(drop=True).copy()
    zeros_to_add_df = aug_df.iloc[zeros_to_add_indices].reset_index(drop=True).copy()

    # back translation
    # await back_translate_samples(zeros_to_add_df, "0")
    # await back_translate_samples(ones_to_add_df, "1")
    print(zeros_to_add_df)
    # synonym replacement
    synonym_replace_samples(zeros_to_add_df, "0")
    # ones_augmented = synonym_replace_samples(ones_to_add_df, "1")
    
    print(zeros_to_add_df)

    # synonym addition

    # easy data augmentation

