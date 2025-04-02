"""
Script to extract positive examples (label == 1) from the training data and save them to a new CSV file.
"""

import pandas as pd
from pathlib import Path

# Path configuration
DATA_DIR = Path(__file__).parent.parent.parent / "data"
TRAIN_FILE = DATA_DIR / "synonym_replacement_all_MiniLM_l6_v2_v2.csv"
POSITIVE_EXAMPLES_FILE = DATA_DIR / "positive_examples.csv"

def extract_positive_examples():
    """Extract positive examples from the training data and save to a new CSV file."""
    # Read the training data
    df = pd.read_csv(TRAIN_FILE)
    
    # Filter for positive examples (label == 1)
    positive_df = df[df['label'] == 1].copy()
    
    # Save to new CSV file
    positive_df.to_csv(POSITIVE_EXAMPLES_FILE, index=False)
    
    # Print statistics
    total_examples = len(df)
    positive_examples = len(positive_df)
    print(f"Total examples in training data: {total_examples}")
    print(f"Positive examples found: {positive_examples}")
    print(f"Percentage of positive examples: {(positive_examples/total_examples)*100:.2f}%")
    print(f"Positive examples saved to: {POSITIVE_EXAMPLES_FILE}")

if __name__ == "__main__":
    extract_positive_examples() 