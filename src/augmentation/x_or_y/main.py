import argparse
import logging
import pandas as pd
import nltk
import random
from tqdm import tqdm
from pathlib import Path
from nltk.corpus import stopwords, wordnet

from src.config import config
from src.augmentation.synonym_replacement.utils import load_cached_embeddings

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

glove_embeddings = load_cached_embeddings()

class XorYAugmenter:
    def __init__(self, train_df: pd.DataFrame, similarity_threshold: float = 0.6, max_choices: int = 2, num_words_to_augment: int = 1):
        self.train_df = train_df
        self.similarity_threshold = similarity_threshold
        self.max_choices = max_choices
        self.num_words_to_augment = num_words_to_augment
        
        self.stop_words = set(stopwords.words('english'))
        self.glove_embeddings = glove_embeddings
        
    
    def _find_candidates(self, claim: str) -> list[tuple[str, str]]:
        """
        Find candidates for X or Y augmentation 

        Args:
            claim (str): The claim to search for candidates

        Returns:
            list[tuple[str, str]]: The possible candidates for augmentation with their POS tags
        """
        tokens = nltk.word_tokenize(claim)
        pos = nltk.pos_tag(tokens)
        
        candidates = []
        
        for word, tag in pos:
            if word.lower() in self.stop_words:
                continue
            
            if word.lower() not in self.glove_embeddings:
                continue
            
            candidates.append((word, tag))
            
        return candidates
                

    def _get_wordnet_pos(self, nltk_tag: str) -> str:
        """
        Map NLTK POS tags to WordNet POS tags.
        """
        tag_map = {
            'JJ': wordnet.ADJ,
            'NN': wordnet.NOUN,
            'VB': wordnet.VERB,
            'RB': wordnet.ADV
        }
        return tag_map.get(nltk_tag[:2], wordnet.NOUN)
        

    def _get_similar_word(self, word: str, pos_tag: str = None) -> list[str]:
        """
        Get similar words using both GloVe embeddings and WordNet

        Args:
            word (str): The word to search for similar words
            pos_tag (str): Part of speech tag

        Returns:
            list[str]: A list of similar words, sorted by similarity
        """     
        topn = max(4, self.max_choices * 3)
        
        candidates = set()
        wordnet_pos = self._get_wordnet_pos(pos_tag) if pos_tag else None

        synsets = wordnet.synsets(word, pos=wordnet_pos)
        if not synsets:
            return []

        for syn in synsets:
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower() and ' ' not in synonym:
                    candidates.add(synonym)
                    if len(candidates) >= topn:
                        break
            if len(candidates) >= topn:
                break

        # Preserve capitalization
        synonym_list = list(candidates)
        if word[0].isupper():
            synonym_list = [s.capitalize() for s in synonym_list]
            
        synonyms_to_return = random.sample(synonym_list, min(topn, len(synonym_list)))

        return synonyms_to_return

    def augment_claims(self, claims: pd.DataFrame) -> pd.DataFrame:
        """
        Augment the claims by adding a '/' between words

        Args:
            claims (pd.DataFrame): The claims to augment

        Returns:
            pd.DataFrame: The augmented claims
        """
        augmented_claims = []
        
        for _, row in tqdm(claims.iterrows(), total=len(claims), desc="Augmenting claims"):
            claim = row['Claim']
            candidates = self._find_candidates(claim)
            
            if len(candidates) == 0:
                continue
            
            # Get the top max(num_words_to_augment, 3) candidates
            candidates = candidates[:max(self.num_words_to_augment, 3)]
            
            # Get a number between 1 and num_words_to_augment
            num_words_to_add = random.randint(1, self.num_words_to_augment)
            
            # Get the top num_words_to_add candidates
            candidates = candidates[:num_words_to_add]
            
            new_claim = claim
            
            for candidate in candidates:
                similar_words = self._get_similar_word(candidate[0], candidate[1])
                
                if len(similar_words) == 0:
                    continue
                
                # Get a random number of words, from 1 to max_choices - 1
                # We minus 1 because we already have the first choice (candidate[0])
                num_words = random.randint(1, self.max_choices - 1)
                
                # Randomly select num_words similar words
                similar_words = random.sample(similar_words, min(num_words, len(similar_words)))
                # Add the original word to the list
                similar_words.append(candidate[0])
                # Randomly shuffle, just in case
                random.shuffle(similar_words)
                        
                new_claim = new_claim.replace(candidate[0], '/'.join(similar_words))
                
            augmented_claims.append({
                "Claim": new_claim,
                "Evidence": row['Evidence'],
                "label": row['label']
            })
                
        return pd.DataFrame(augmented_claims)
                

def main():
    parser = argparse.ArgumentParser(
        description='Data augmention by adding / to claims'
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
        help='Threshold for similarity between word and its other words'
    )
    args = parser.parse_args()
    
    output_path = config.DATA_DIR / Path(args.output_file)
    
    train_df = pd.read_csv(config.TRAIN_FILE)
    train_df = train_df[train_df['label'] == 1]
    
    augmenter = XorYAugmenter(train_df, args.similarity_threshold, args.max_choices, args.num_words_to_augment)
    
    augmented_df = augmenter.augment_claims(train_df)
    
    augmented_df.to_csv(output_path, index=False)
    

    
    
    
