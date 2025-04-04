import asyncio
import logging
import pandas as pd
from pathlib import Path
from googletrans import Translator
from typing import AsyncGenerator


# Set up logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

# Import config
from src.config import config


def format_text(text: str) -> str:
    # Replace all double quotes with single quotes
    text = text.replace('\"', "\'")
    
    # If the text starts with a double quote, remove it
    text = text[1:] if text.startswith('"') else text

    # If the text ends with a double quote, remove it
    text = text[:-1] if text.endswith('"') else text
    
    # If the text contains a commma, or a single quote, wrap it in double quotes
    text = f'"{text}"' if ',' in text or "'" in text else text
        
    # If the text contains unrepresentable characters, replace them with a space
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    return text


async def back_translate(text: str, src='en', intermediate='fr') -> str:
    async with Translator() as translator:
        to_french = await translator.translate(text, src=src, dest=intermediate, )
        back_to_english = await translator.translate(
            to_french.text, 
            src=intermediate, 
            dest=src
        )
        return back_to_english.text
    
async def back_translate_batch(data: pd.DataFrame, column: str, src='en', intermediate='fr') -> pd.DataFrame:
    async with Translator() as translator:
        # Apply translation to each element in the specified column
        async def translate_text(text):
            translation = await translator.translate(text, src=src, dest=intermediate)
            back_to_english = await translator.translate(translation.text, src=intermediate, dest=src)
            return back_to_english.text
        
        # Apply the translation function to the DataFrame column
        if column == "Both":
            data.loc[:, "Claim"] = [await translate_text(text) for text in data["Claim"]]
            data.loc[:, "Evidence"] = [await translate_text(text) for text in data["Evidence"]]
        else:
            data.loc[:, column] = [await translate_text(text) for text in data[column]]

        return data

async def process_data_stream(train_path: Path, dev_path: Path, augmented_data: pd.DataFrame) -> AsyncGenerator[dict, None]:
    train_df = pd.read_csv(train_path)
    positive_examples = train_df[train_df['label'] == 1]
    
    for idx, row in positive_examples.iterrows():
        if row['Evidence'] in augmented_data['Original Evidence'].values:
            # Skip this example as it's already been processed
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
    augmented_data = pd.read_csv(config.AUG_TRAIN_FILE)

    async for item in process_data_stream(config.TRAIN_FILE, config.DEV_FILE, augmented_data):
        translated_evidence = format_text(item['translated_evidence'])
        original_evidence = format_text(item['original_evidence'])
        claim = format_text(item['claim'])
        
        with open(config.AUG_TRAIN_FILE, 'a') as f:
            f.write(f'{claim},{translated_evidence},{original_evidence},1\n')
            
        # Log the augmented data
        logging.info(f"Added augmented data for claim: {claim[:30]}...")
        
        # Save to the augmented dataframe as well
        new_row = pd.DataFrame({
            'Claim': [claim], 
            'Evidence': [translated_evidence], 
            'Original Evidence': [original_evidence],
            'label': [1]
        })
        augmented_data = pd.concat([augmented_data, new_row], ignore_index=True)


if __name__ == "__main__":
    asyncio.run(main())
