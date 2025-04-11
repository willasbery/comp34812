import argparse
import logging
import random
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet
from tqdm import tqdm

from src.config import config
from src.augmentation.synonym_replacement.utils import load_cached_embeddings

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

# Load embeddings once for reuse
glove_embeddings = load_cached_embeddings()


class XorYAugmenter:
    """
    A text augmentation class that replaces words with alternatives in X/Y format.
    
    This augmenter finds candidate words in a text and replaces them with a format
    like "word/synonym1/synonym2", creating variations of the original text.
    """
    
    def __init__(self, train_df: pd.DataFrame, similarity_threshold: float = 0.6, 
                 max_choices: int = 2, num_words_to_augment: int = 1):
        """
        Initialize the XorYAugmenter.
        
        Args:
            train_df: Training dataframe containing text to analyze
            similarity_threshold: Threshold for word similarity (0.0-1.0)
            max_choices: Maximum number of alternative words to include
            num_words_to_augment: Number of words to augment in each text
        """
        self.train_df = train_df
        self.similarity_threshold = similarity_threshold
        self.max_choices = max_choices
        self.num_words_to_augment = num_words_to_augment
        
        self.stop_words = set(stopwords.words('english'))
        self.glove_embeddings = glove_embeddings
    
    def _find_candidates(self, claim: str) -> list[tuple[str, str]]:
        """
        Find candidate words for augmentation in the given text.
        
        Args:
            claim: The text to analyze for augmentation candidates
            
        Returns:
            List of (word, POS tag) tuples that are candidates for augmentation
        """
        tokens = nltk.word_tokenize(claim)
        pos = nltk.pos_tag(tokens)
        
        candidates = []
        
        for word, tag in pos:
            # Skip stopwords and words not in our embedding vocabulary
            if word.lower() in self.stop_words or word.lower() not in self.glove_embeddings:
                continue
            
            candidates.append((word, tag))
            
        return candidates
                
    def _get_wordnet_pos(self, nltk_tag: str) -> str:
        """
        Map NLTK POS tags to WordNet POS tags.
        
        Args:
            nltk_tag: POS tag from NLTK tagger
            
        Returns:
            Corresponding WordNet POS tag
        """
        tag_map = {
            'JJ': wordnet.ADJ,
            'NN': wordnet.NOUN,
            'VB': wordnet.VERB,
            'RB': wordnet.ADV,
            'MD': wordnet.VERB
        }
        return tag_map.get(nltk_tag[:2], wordnet.NOUN)
        
    def _get_similar_words(self, word: str, pos_tag: str = None) -> list[str]:
        """
        Find similar words using WordNet.
        
        Args:
            word: The word to find synonyms for
            pos_tag: Part of speech tag to constrain synonyms
            
        Returns:
            List of similar words suitable for augmentation
        """     
        topn = max(4, self.max_choices * 3)
        
        candidates = set()
        wordnet_pos = self._get_wordnet_pos(pos_tag) if pos_tag else None

        synsets = wordnet.synsets(word, pos=wordnet_pos)
        if not synsets:
            return []

        # Collect synonyms from WordNet
        for syn in synsets:
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower() and ' ' not in synonym:
                    candidates.add(synonym)
                    if len(candidates) >= topn:
                        break
            if len(candidates) >= topn:
                break

        # Preserve original capitalization
        synonym_list = list(candidates)
        if word[0].isupper():
            synonym_list = [s.capitalize() for s in synonym_list]
            
        # Sample a subset of synonyms
        synonyms_to_return = random.sample(synonym_list, min(topn, len(synonym_list)))

        return synonyms_to_return
    
    def _augment_text(self, text: str) -> str | None:
        """
        Augment a single text by replacing words with X/Y alternatives.
        
        Args:
            text: The text to augment
            
        Returns:
            Augmented text or None if augmentation was not possible
        """
        candidates = self._find_candidates(text)
        if not candidates:
            return None

        # Determine the number of candidates to use
        num_candidates = min(len(candidates), random.randint(1, self.num_words_to_augment))
        candidates = candidates[:num_candidates]

        for candidate in candidates:
            similar_words = self._get_similar_words(candidate[0], candidate[1])
            if not similar_words:
                continue

            # Select a random number of similar words
            num_words = min(len(similar_words), random.randint(1, self.max_choices - 1))
            similar_words = random.sample(similar_words, num_words)
            similar_words.append(candidate[0])
            random.shuffle(similar_words)

            text = text.replace(candidate[0], '/'.join(similar_words))

        return text

    def augment_data(self, data: pd.DataFrame, augment_claim: bool = True, augment_evidence: bool = False) -> None:
        """
        Augment a dataset by adding X/Y alternatives to selected fields.
        
        This method modifies the dataframe in-place, adding alternatives to either
        claims, evidence, or both depending on the parameters.
        
        Args:
            data: DataFrame containing text to augment
            augment_claim: Whether to augment the 'Claim' column
            augment_evidence: Whether to augment the 'Evidence' column
        """
        for index, row in tqdm(data.iterrows(), total=len(data), desc="Augmenting dataset"):
            if augment_claim:
                new_claim = self._augment_text(row['Claim'])
                if new_claim:
                    data.at[index, 'Claim'] = new_claim
            
            if augment_evidence:
                new_evidence = self._augment_text(row['Evidence'])
                if new_evidence:
                    data.at[index, 'Evidence'] = new_evidence


def main():
    """
    Main function to run the X/Y augmentation on the dataset.
    
    Parses command line arguments and runs the augmentation process.
    """
    parser = argparse.ArgumentParser(
        description='Data augmentation by adding X/Y alternatives to texts'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='x_or_y_augmented_claims.csv',
        help='Path to save the augmented claims'
    )
    parser.add_argument(
        '--max_choices',
        type=int,
        default=2,
        help='Maximum number of words in /, i.e., w/x/y/z is 4'
    )
    parser.add_argument(
        '--num_words_to_augment',
        type=int,
        default=1,
        help='Number of words to augment in the claim'
    )
    parser.add_argument(
        '--similarity_threshold',
        type=float,
        default=0.6,
        help='Threshold for similarity between word and its alternatives'
    )
    args = parser.parse_args()
    
    output_path = config.DATA_DIR / Path(args.output_file)
    
    # Load and filter training data
    train_df = pd.read_csv(config.TRAIN_FILE)
    train_df = train_df[train_df['label'] == 1]
    
    # Create augmenter and process data
    augmenter = XorYAugmenter(
        train_df, 
        args.similarity_threshold, 
        args.max_choices, 
        args.num_words_to_augment
    )
    
    augmenter.augment_data(train_df)
    
    # Save augmented data
    train_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
    

    
    
