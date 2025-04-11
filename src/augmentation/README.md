# Data Augmentation Documentation

This directory contains various data augmentation techniques implemented for the task. The augmentation pipeline combines multiple techniques to generate diverse training data while preserving the semantic meaning of the original text.

## Augmentation Techniques

### 1. Back Translation (`back_translation/`)

Translates text to an intermediate language and back to English to generate paraphrased versions while preserving meaning.

- **Features:**

  - Supports multiple intermediate languages (French, German, Japanese)
  - Configurable translation splits (Claim, Evidence, or Both)
  - Handles text formatting and special characters
  - Includes retry mechanism for failed translations

- **Usage:**

  ```python
  from src.augmentation.back_translation.main import back_translate_batch

  # Translate a batch of data
  augmented_data = await back_translate_batch(
      data=dataframe,
      column="Both",  # or "Claim" or "Evidence"
      src="en",
      intermediate="fr"
  )
  ```

### 2. Synonym Replacement (`synonym_replacement/`)

Replaces words with their synonyms using both GloVe embeddings and WordNet.

- **Features:**

  - Uses GloVe embeddings for semantic similarity
  - Considers word context and part-of-speech
  - Configurable similarity thresholds
  - Caches embeddings for faster processing

- **Key Components:**
  - `utils.py`: Helper functions for text processing
  - `all_MiniLM_l6_v2/`: Advanced synonym replacement using sentence embeddings
  - `original/`: Basic synonym replacement implementation

### 3. X or Y Augmentation (`x_or_y/`)

Adds alternative words using the "/" notation (e.g., "leads to" → "leads to/results in/causes").

- **Features:**

  - Preserves word capitalization
  - Considers part-of-speech tags
  - Configurable number of alternatives
  - Uses both GloVe and WordNet for synonym selection

- **Usage:**

  ```python
  from src.augmentation.x_or_y.main import XorYAugmenter

  augmenter = XorYAugmenter(
      train_df=dataframe,
      similarity_threshold=0.6,
      max_choices=2,
      num_words_to_augment=1
  )
  augmenter.augment_data(dataframe)
  ```

### 4. Pipeline (`pipeline/`)

Combines all augmentation techniques into a unified pipeline.

- **Features:**

  - Configurable augmentation percentages per label
  - Supports both replacement and addition strategies
  - Handles data balancing
  - Detailed logging of augmentation process

- **Configuration:**
  ```python
  AUGMENTATION_CONFIG = {
      "0": {  # Negative examples
          "replace": 0.2,  # Replace 20% of negative examples
          "add": 0.1,      # Add 10% more negative examples
          "translate": {
              "percentage": 0.5,
              "split": {"Claim": 0.3, "Evidence": 0.3, "Both": 0.4},
              "intermediates": {"fr": 0.4, "de": 0.3, "es": 0.3}
          },
          "synonym_replacement": {
              "percentage": 0.4,
              "replacement_fraction": 0.2,
              "min_similarity": 0.7
          },
          "x_or_y": {
              "percentage": 0.3,
              "split": {"Claim": 0.4, "Evidence": 0.4, "Both": 0.2},
              "max_choices": 2
          }
      },
      "1": {  # Positive examples
          # Similar configuration for positive examples
      }
  }
  ```

## Usage Example

```python
from src.augmentation.pipeline.main import main

# Run the complete augmentation pipeline
await main()
```

## Best Practices

1. **Data Quality**

   - Monitor the quality of augmented data
   - Validate semantic preservation
   - Check for introduced noise or errors

2. **Configuration**

   - Adjust augmentation percentages based on data distribution
   - Balance between diversity and quality
   - Consider computational resources

3. **Performance**
   - Use caching for embeddings
   - Implement batch processing
   - Monitor memory usage

## Dependencies

- Python 3.8+
- pandas
- NLTK
- Gensim
- googletrans
- tqdm
- numpy
