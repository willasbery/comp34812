"""
Back translation module for data augmentation.

This module implements back translation, a technique that translates text to an
intermediate language and then back to the original language to create paraphrased
versions of the original text, used for data augmentation.
"""
import asyncio
import logging
import pandas as pd
from pathlib import Path
from googletrans import Translator
from typing import AsyncGenerator


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

from src.config import config


def format_text(text: str) -> str:
    """
    Format text to ensure compatibility with CSV format and remove problematic characters.
    
    This function:
    - Replaces double quotes with single quotes
    - Removes leading/trailing quotes
    - Wraps text in quotes if it contains commas or single quotes
    - Removes non-ASCII characters
    
    Args:
        text: Input text to be formatted
        
    Returns:
        Formatted text string ready for CSV storage
    """
    text = text.replace('\"', "\'")
    text = text[1:] if text.startswith('"') else text
    text = text[:-1] if text.endswith('"') else text
    text = f'"{text}"' if ',' in text or "'" in text else text
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    return text


async def back_translate(text: str, src='en', intermediate='fr') -> str:
    """
    Translate text to an intermediate language and back to the source language.
    
    Args:
        text: Text to be back-translated
        src: Source language code
        intermediate: Intermediate language code
        
    Returns:
        Back-translated text
    """
    async with Translator() as translator:
        to_intermediate = await translator.translate(text, src=src, dest=intermediate)
        back_to_source = await translator.translate(
            to_intermediate.text, 
            src=intermediate, 
            dest=src
        )
        return back_to_source.text


async def back_translate_batch(data: pd.DataFrame, column: str, src='en', intermediate='fr') -> pd.DataFrame:
    """
    Apply back-translation to a batch of text in a DataFrame.
    
    Args:
        data: DataFrame containing the text to translate
        column: Column name to translate (or "Both" for "Claim" and "Evidence")
        src: Source language code
        intermediate: Intermediate language code
        
    Returns:
        DataFrame with translated text
    """
    async with Translator() as translator:
        async def translate_text(text):
            """
            Translate a single text with retry mechanism.
            
            Makes up to 3 attempts to translate the text, waiting 0.5s between attempts.
            Returns original text if all attempts fail.
            """
            for attempt in range(3):
                try:
                    translation = await translator.translate(text, src=src, dest=intermediate)
                    back_to_source = await translator.translate(translation.text, src=intermediate, dest=src)
                    return back_to_source.text
                except Exception as e:
                    logging.warning(f"Attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(0.5)
            return text
        
        if column == "Both":
            data.loc[:, "Claim"] = [await translate_text(text) for text in data["Claim"]]
            data.loc[:, "Evidence"] = [await translate_text(text) for text in data["Evidence"]]
        else:
            data.loc[:, column] = [await translate_text(text) for text in data[column]]

        return data


async def process_data_stream(train_path: Path, dev_path: Path, augmented_data: pd.DataFrame) -> AsyncGenerator[dict, None]:
    """
    Process training data and generate augmented examples via back-translation.
    
    Reads positive examples from training data and yields back-translated versions
    for those that haven't been processed yet.
    
    Args:
        train_path: Path to the training data file
        dev_path: Path to the development data file
        augmented_data: DataFrame containing already augmented data
        
    Yields:
        Dictionary with original and translated text pairs
    """
    train_df = pd.read_csv(train_path)
    positive_examples = train_df[train_df['label'] == 1]
    
    for idx, row in positive_examples.iterrows():
        if row['Evidence'] in augmented_data['Original Evidence'].values:
            logging.info(f"Skipping example {idx} as it's already been processed")
            continue
        
        original_evidence = row['Evidence']
        translated_evidence = await back_translate(original_evidence)
        
        yield {
            'claim': row['Claim'],
            'original_evidence': original_evidence,
            'translated_evidence': translated_evidence,
            'index': idx
        }


async def main():
    """
    Main function to process and augment data using back-translation.
    
    Reads existing augmented data, processes unaugmented examples,
    and appends newly augmented data to the output file and DataFrame.
    Each augmented example is formatted for CSV storage and logged.
    """
    augmented_data = pd.read_csv(config.AUG_TRAIN_FILE)

    async for item in process_data_stream(config.TRAIN_FILE, config.DEV_FILE, augmented_data):
        translated_evidence = format_text(item['translated_evidence'])
        original_evidence = format_text(item['original_evidence'])
        claim = format_text(item['claim'])
        
        with open(config.AUG_TRAIN_FILE, 'a') as f:
            f.write(f'{claim},{translated_evidence},{original_evidence},1\n')
            
        logging.info(f"Added augmented data for claim: {claim[:30]}...")
        
        new_row = pd.DataFrame({
            'Claim': [claim], 
            'Evidence': [translated_evidence], 
            'Original Evidence': [original_evidence],
            'label': [1]
        })
        augmented_data = pd.concat([augmented_data, new_row], ignore_index=True)


if __name__ == "__main__":
    asyncio.run(main())
