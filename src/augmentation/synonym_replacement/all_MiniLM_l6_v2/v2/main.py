import argparse
import datetime
import json
import logging
import nltk
import numpy as np
import pandas as pd
import random
import re
import string
from collections import defaultdict, Counter
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import wordnet, stopwords

from src.config import config
from src.utils.utils import get_device
from src.augmentation.synonym_replacement.utils import remove_stopwords

import warnings
warnings.filterwarnings('ignore')

nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)


class AdvancedSynonymReplacer:
    """
    An enhanced class to perform synonym replacement-based data augmentation with more options and control.

    Attributes:
        params (dict): Dictionary of user-defined parameters.
        train_df (pd.DataFrame): DataFrame containing 'Evidence', 'Claim', and 'label' columns.
        device (str): Torch device (e.g., 'cpu', 'cuda').
        st_model (SentenceTransformer): Loaded Sentence Transformer model for semantic similarity.
        stop_words (set): Set of English stop words.
        word_frequencies (Counter): Frequency of words in the training data.
    """

    def __init__(self, params: dict, train_df: pd.DataFrame):
        """
        Initialise the AdvancedSynonymReplacer with parameters and training data.

        Args:
            params (dict): Dictionary of parameters for augmentation.
            train_df (pd.DataFrame): Original training DataFrame.
        """
        self.params = params
        self.device = get_device()
        self.stop_words = set(stopwords.words('english'))

        # Load Sentence Transformer Model
        self.st_model_name = params.get('sentence_transformer_model', 'sentence-transformers/all-MiniLM-L6-v2')
        logging.info(f"Loading Sentence Transformer model: {self.st_model_name} onto device: {self.device}")
        self.st_model = SentenceTransformer(self.st_model_name, device=self.device)
        logging.info("Sentence Transformer model loaded.")

        # Parameters
        self.min_sentence_similarity = params.get("min_sentence_similarity", 0.85)
        self.replacement_fraction = params.get("replacement_fraction", 0.5)
        self.batch_size = params.get("batch_size", 1000)
        self.add_original_evidence_to_results = params.get("add_original_evidence_to_results", True)
        self.results_file_name = params.get("output_file", config.DATA_DIR / "advanced_synonym_replacement_results.csv")
        self.min_word_length = params.get("min_word_length", 4)
        self.synonym_selection_strategy = params.get("synonym_selection_strategy", "random")  # 'random' or 'frequent'
        self.allow_multi_word_synonyms = params.get("allow_multi_word_synonyms", False)
        self.word_frequency_threshold = params.get("word_frequency_threshold", 5) # Minimum frequency for a word to be considered for replacement
        
        self.enable_random_synonym_insertion = params.get("enable_random_synonym_insertion", False)
        self.synonym_insertion_probability = params.get("synonym_insertion_probability", 0.05)
        
        self.enable_random_word_insertion = params.get("enable_random_word_insertion", False)
        self.word_insertion_probability = params.get("word_insertion_probability", 0.05)
        
        self.enable_random_deletion = params.get("enable_random_synonym_deletion", False)
        self.deletion_probability = params.get("deletion_probability", 0.05)

        # Store original DataFrame and create POS tags
        self.train_df = train_df.copy()
        self._prepare_data()

        logging.info("Starting advanced data augmentation with the following parameters:")
        for key, value in params.items():
            logging.info(f" - {key}: {value}")


    def _prepare_data(self):
        """Prepares the data by adding POS tags and calculating word frequencies."""
        if 'POS' not in self.train_df.columns:
            self.train_df['POS_Evidence'] = self.train_df['Evidence'].apply(
                lambda x: nltk.pos_tag(nltk.word_tokenize(x))
            )

        self.original_evidences_pos = self.train_df['POS_Evidence'].tolist()
        self.original_evidences = self.train_df['Evidence'].tolist()
        self.preprocessed_evidences = self.train_df['Evidence'].apply(remove_stopwords).tolist()
        self.corresponding_claim = self.train_df['Claim'].apply(remove_stopwords).tolist()

        # Calculate word frequencies
        all_words = []
        for text in self.train_df['Evidence']:
            all_words.extend(nltk.word_tokenize(text.lower()))
        self.word_frequencies = Counter(all_words)


    def calculate_sentence_similarity(self, sentence_1: str, sentence_2: str) -> float:
        """
        Calculate semantic similarity between two sentences using Sentence Transformers.

        Args:
            sentence_1 (str): The first sentence.
            sentence_2 (str): The second sentence.

        Returns:
            float: Semantic similarity score between 0 and 1.
        """
        embeddings = self.st_model.encode([sentence_1, sentence_2], convert_to_tensor=True, device=self.device, verbose=False)
        cosine_scores = util.cos_sim(embeddings[0], embeddings[1])
        return cosine_scores.item()


    def _get_wordnet_pos(self, tag):
        """Map NLTK POS tags to WordNet POS tags."""
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None


    def _process_text(
        self,
        text_tokens: list[str],
        pos_tags: list[tuple[str, str]],
        claim_words: set = None,
        is_claim: bool = False
    ) -> list[str]:
        """
        Identify potential candidate words in the text for replacement.

        Args:
            text_tokens (list[str]): Tokenised words (original form) from the text.
            pos_tags (list[tuple[str, str]]): List of (word, POS tag) tuples.
            claim_words (set, optional): Set of words in the claim (preprocessed). Defaults to None.
            is_claim (bool, optional): Whether the text being processed is a claim. Defaults to False.

        Returns:
            list[str]: List of tokens deemed eligible for synonym replacement.
        """
        potential_replacements = []
        safe_pos_tags = {
            'NN', 'NNS', 'JJ', 'JJR', 'JJS',
            'RB', 'RBR', 'RBS',
            'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'
        }

        pos_tags_dict = defaultdict(list)
        for word, tag in pos_tags:
            pos_tags_dict[word.lower()].append(tag)

        common_words = set()
        if claim_words is not None and not is_claim:
            common_words = set(text_tokens) & claim_words

        for word in text_tokens:
            lower_word = word.lower()

            # Skip if stop word
            if lower_word in self.stop_words:
                continue

            # Skip based on frequency
            if self.word_frequencies.get(lower_word, 0) < self.word_frequency_threshold:
                continue

            # Skip if:
            # 1. The word is in both claim and evidence (if applicable and not processing claim)
            # 2. The word is a substring of any claim word (or vice versa) (if applicable and not processing claim)
            # 3. The word not in pos_tags_dict
            # 4. The word is too short
            # 5. The word's POS tags are not in the safe list
            if (not is_claim and (word in common_words or
                    any(word in cw or cw in word for cw in claim_words))) or \
               lower_word not in pos_tags_dict or \
               len(word) < self.min_word_length or \
               not any(tag in safe_pos_tags for tag in pos_tags_dict[lower_word]):
                continue
            potential_replacements.append(word)

        return potential_replacements


    def get_synonyms(self, word: str, pos_tag: str = None, topn: int = 10) -> list[str]:
        """
        Retrieve synonyms for a given word using the NLTK WordNet corpus.

        Args:
            word (str): The target word to find synonyms for.
            pos_tag (str): The NLTK POS tag of the word (e.g., 'NN', 'VB', etc.).
            topn (int): Maximum number of synonyms to return.

        Returns:
            list[str]: List of potential synonyms.
        """
        synonyms = set()
        wordnet_pos = self._get_wordnet_pos(pos_tag) if pos_tag else None

        synsets = wordnet.synsets(word, pos=wordnet_pos)
        if not synsets:
            return []

        for syn in synsets:
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower():
                    if self.allow_multi_word_synonyms:
                        synonyms.add(synonym)
                    elif ' ' not in synonym:
                        synonyms.add(synonym)
                        
                    if len(synonyms) >= topn:
                        break
                    
            if len(synonyms) >= topn:
                break

        synonym_list = list(synonyms)
        if self.synonym_selection_strategy == 'frequent':
            # Try to prioritize synonyms that appear more frequently in the training data
            synonym_list.sort(key=lambda s: self.word_frequencies.get(s.lower(), 0), reverse=True)
        elif self.synonym_selection_strategy == 'random':
            random.shuffle(synonym_list)

        return synonym_list[:topn]

    def get_random_word(self) -> list[str]:
        """
        Get a random word from the training data.

        Returns:
            str: A random word from the training data.
        """
        return [random.choice(list(self.word_frequencies.keys()))]


    def find_valid_replacements(
        self,
        word_to_replace: str,
        synonyms: list[str],
        original_text: str,
        original_pos_tags: dict
    ) -> tuple[bool, str]:
        """
        From a list of synonyms, find a valid synonym replacement that maintains
        the POS tag and ensures the final augmented sentence remains semantically
        similar to the original sentence.

        Args:
            word_to_replace (str): The original word to be replaced.
            synonyms (list[str]): List of potential synonym strings.
            original_text (str): The unmodified text.
            original_pos_tags (dict): Dictionary mapping words (lowercased) to their POS tags.

        Returns:
            tuple[bool, str]: (True, replacement_word) if a valid replacement is found, else (False, "").
        """
        lower_word = word_to_replace.lower()
        original_word_pos_tags = original_pos_tags.get(lower_word, [])
        if not original_word_pos_tags:
            return False, ""
        primary_pos = original_word_pos_tags[0]

        for synonym in synonyms:
            pattern = r'\b' + re.escape(word_to_replace) + r'\b'
            try:
                new_text = re.sub(pattern, synonym, original_text, flags=re.IGNORECASE)
            except re.error:
                logging.warning(f"Regex error replacing '{word_to_replace}' with '{synonym}'. Skipping.")
                continue

            if new_text == original_text:
                continue

            new_text_pos = nltk.pos_tag(nltk.word_tokenize(new_text))
            synonym_pos_tags = [
                tag for (w, tag) in new_text_pos if w.lower() == synonym.lower()
            ]
            if not synonym_pos_tags:
                continue

            # Basic POS check
            if self._get_wordnet_pos(synonym_pos_tags[0]) != self._get_wordnet_pos(primary_pos):
                continue

            # Check semantic similarity
            similarity = self.calculate_sentence_similarity(original_text, new_text)
            if similarity >= self.min_sentence_similarity:
                return True, synonym

        return False, ""


    def _random_insertion(self, tokens: list[str], pos_tags: list[tuple[str, str]], add_a_synonym: bool = True) -> list[str]:
        """Randomly insert synonyms into the text."""
        augmented_tokens = list(tokens)
        
        if not augmented_tokens:
            return []
        
        insert_index = random.randint(0, len(augmented_tokens))
        word_to_augment = random.choice(augmented_tokens)
        lower_word = word_to_augment.lower()
        original_word_pos_tags = dict(pos_tags).get(lower_word, [])
        
        if not original_word_pos_tags:
            return []
        
        if add_a_synonym:
            synonyms = self.get_synonyms(word_to_augment, original_word_pos_tags, topn=5)
        else:
            synonyms = self.get_random_word()
            
        if not synonyms:
            return []
        
        synonym_to_insert = random.choice(synonyms)
        augmented_tokens.insert(insert_index, synonym_to_insert)
                    
        return augmented_tokens


    def _random_deletion(self, tokens: list[str]) -> list[str]:
        """Randomly delete words from the text."""
        if not tokens:
            return []
        
        augmented_tokens = [token for token in tokens]  
        indices_to_delete = random.sample(range(len(augmented_tokens)), random.randint(0, int(len(augmented_tokens) * 0.1)))
        augmented_tokens = [token for i, token in enumerate(augmented_tokens) if i not in indices_to_delete]
        
        return augmented_tokens


    def augment_data(self):
        """
        Perform the data augmentation on each evidence and optionally claim
        in the training DataFrame by replacing words with synonyms, and optionally
        inserting or deleting words. The augmented data is saved in CSV batches.
        """
        results_path = Path(self.results_file_name)
        if results_path.exists():
            overwrite = input(
                f"Results file {self.results_file_name} already exists. Overwrite? (y/n) "
            ).strip().lower()
            if overwrite != 'y':
                logging.info("Augmentation aborted by user.")
                return
        else:
            results_path.parent.mkdir(parents=True, exist_ok=True)

        cols = ["Claim", "Evidence", "label"]
        if self.add_original_evidence_to_results:
            cols.append("Original Evidence")
            cols.append("Similarity Score")

        synonym_replaced_df = pd.DataFrame(columns=cols)
        batch_counter = 0

        successful_augmentations = 0
        attempted_augmentations = 0
        total_words_augmented = 0  # Counter for the number of words augmented

        original_claims = self.train_df['Claim'].tolist()
        labels = self.train_df['label'].tolist()

        for idx, original_evidence_text in tqdm(
            enumerate(self.original_evidences),
            desc="Augmenting data",
            total=len(self.original_evidences)
        ):
            attempted_augmentations += 1
            original_claim_text = original_claims[idx]

            # POS tagging for evidence
            evidence_pos_tags = self.original_evidences_pos[idx]
            evidence_pos_tags_dict = defaultdict(list)
            for word, tag in evidence_pos_tags:
                evidence_pos_tags_dict[word.lower()].append(tag)
            evidence_tokens = nltk.word_tokenize(original_evidence_text)

            augmented_evidence_tokens = list(evidence_tokens)
            are_synonyms_inserted = random.randint(0, 100) <= int((100 * self.synonym_insertion_probability))
            if self.enable_random_synonym_insertion and are_synonyms_inserted:
                augmented_evidence_tokens = self._random_insertion(augmented_evidence_tokens, evidence_pos_tags, add_a_synonym=True)
                
            are_words_inserted = random.randint(0, 100) <= int((100 * self.word_insertion_probability))
            if self.enable_random_word_insertion and are_words_inserted:
                augmented_evidence_tokens = self._random_insertion(augmented_evidence_tokens, evidence_pos_tags, add_a_synonym=False)
                
            are_synonyms_deleted = random.randint(0, 100) <= int((100 * self.deletion_probability))
            if self.enable_random_deletion and are_synonyms_deleted:
                augmented_evidence_tokens = self._random_deletion(augmented_evidence_tokens)

            potential_evidence_replacements = self._process_text(
                augmented_evidence_tokens,
                evidence_pos_tags,
                claim_words=set()
            )

            num_evidence_replacements = max(0, int(len(potential_evidence_replacements) * self.replacement_fraction))
            
            if potential_evidence_replacements and num_evidence_replacements > 0:
                words_to_replace = random.sample(potential_evidence_replacements, k=num_evidence_replacements)
                current_evidence = " ".join(augmented_evidence_tokens)
                final_word_replacement_map_evidence = {}

                for word in words_to_replace:
                    lower_word = word.lower()
                    if lower_word not in evidence_pos_tags_dict or not evidence_pos_tags_dict[lower_word]:
                        continue

                    word_pos_tag = evidence_pos_tags_dict[lower_word][0]
                    synonyms = self.get_synonyms(word, word_pos_tag, topn=10)
                    if not synonyms:
                        continue

                    found, replacement = self.find_valid_replacements(
                        word,
                        synonyms,
                        current_evidence,
                        evidence_pos_tags_dict
                    )
                    if found:
                        pattern = r'\b' + re.escape(word) + r'\b'
                        try:
                            current_evidence = re.sub(pattern, replacement, current_evidence, flags=re.IGNORECASE)
                            final_word_replacement_map_evidence[word] = replacement
                            total_words_augmented += 1  # Increment the counter for each successful replacement
                        except re.error:
                            logging.warning(f"Regex error applying replacement for '{word}' with '{replacement}'.")
                augmented_evidence_text = current_evidence
            else:
                augmented_evidence_text = " ".join(augmented_evidence_tokens)
                
            # Remove spaces around hyphens and ensure no space after '<', '[', '(', '{'
            excluded_punctuation = "<([{"
            punctuation_to_check = ''.join(c for c in string.punctuation if c not in excluded_punctuation)
            augmented_evidence_text = re.sub(r'\s+([{}])'.format(re.escape(punctuation_to_check)), r'\1', augmented_evidence_text)
            augmented_evidence_text = re.sub(r'\s*-\s*', '-', augmented_evidence_text)  # Handle hyphens
            augmented_evidence_text = re.sub(r'([<\[({])\s+', r'\1', augmented_evidence_text)  # No space after '<', '[', '(', '{'

            # Validate the final augmented evidence text
            final_similarity_score = self.calculate_sentence_similarity(original_evidence_text, augmented_evidence_text)
            if final_similarity_score >= self.min_sentence_similarity:
                # Construct the new row
                new_row_data = {
                    "Claim": original_claim_text,
                    "Evidence": augmented_evidence_text,
                    "label": labels[idx]
                }
                
                if self.add_original_evidence_to_results:
                    new_row_data["Original Evidence"] = original_evidence_text
                    new_row_data["Similarity Score"] = final_similarity_score

                synonym_replaced_df = pd.concat([
                    synonym_replaced_df,
                    pd.DataFrame([new_row_data])
                ], ignore_index=True)
                successful_augmentations += 1

            # Save batch if it reaches batch size
            if len(synonym_replaced_df) >= self.batch_size:
                mode = 'w' if batch_counter == 0 else 'a'
                header = (batch_counter == 0)
                logging.info(f"Saving batch {batch_counter} to {self.results_file_name}")
                synonym_replaced_df.to_csv(self.results_file_name, index=False, mode=mode, header=header)
                synonym_replaced_df = pd.DataFrame(columns=cols)
                batch_counter += 1

        # Save any remaining augmented data
        if not synonym_replaced_df.empty:
            mode = 'w' if batch_counter == 0 else 'a'
            header = (batch_counter == 0)
            synonym_replaced_df.to_csv(self.results_file_name, index=False, mode=mode, header=header)

        # Log final statistics
        logging.info(f"Augmentation completed. {successful_augmentations} sentences successfully augmented "
                     f"out of {attempted_augmentations} attempts.")
        logging.info(f"Total words augmented: {total_words_augmented}")
        if attempted_augmentations > 0:
            rate = (successful_augmentations / attempted_augmentations) * 100
            logging.info(f"Success rate: {rate:.2f}%")
        else:
            logging.info("No augmentation attempts were made.")

        return successful_augmentations


class AdvancedSynonymReplacerDF(AdvancedSynonymReplacer):
    """
    A variation of AdvancedSynonymReplacer that modifies the input DataFrame directly
    and preserves stop words for use with transformer models.
    """
    
    def __init__(self, params: dict, train_df: pd.DataFrame):
        """
        Initialize the AdvancedSynonymReplacerDF with parameters and training data.
        
        Args:
            params (dict): Dictionary of parameters for augmentation.
            train_df (pd.DataFrame): Original training DataFrame.
        """
        super().__init__(params, train_df)
        self.train_df = train_df
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepares the data by adding POS tags and calculating word frequencies without removing stopwords."""
        if 'POS' not in self.train_df.columns:
            self.train_df['POS_Evidence'] = self.train_df['Evidence'].apply(
                lambda x: nltk.pos_tag(nltk.word_tokenize(x))
            )

        # Calculate word frequencies
        all_words = []
        for text in self.train_df['Evidence']:
            all_words.extend(nltk.word_tokenize(text.lower()))
        self.word_frequencies = Counter(all_words)
    
    def augment_data(self):
        """
        Perform the data augmentation on each evidence in the input DataFrame
        by replacing words with synonyms, and optionally inserting or deleting words.
        Modifies the input DataFrame in-place.
        
        Returns:
            pd.DataFrame: Reference to the modified input DataFrame.
        """
        successful_augmentations = 0
        attempted_augmentations = 0
        total_words_augmented = 0  # Counter for the number of words augmented

        for idx, row in tqdm(
            self.train_df.iterrows(),
            desc="Augmenting data",
            total=len(self.train_df)
        ):
            attempted_augmentations += 1
            original_evidence_text = row['Evidence']
            original_claim_text = row['Claim']

            # POS tagging for evidence
            evidence_pos_tags = row['POS_Evidence']
            evidence_pos_tags_dict = defaultdict(list)
            for word, tag in evidence_pos_tags:
                evidence_pos_tags_dict[word.lower()].append(tag)
            evidence_tokens = nltk.word_tokenize(original_evidence_text)

            augmented_evidence_tokens = list(evidence_tokens)
            are_synonyms_inserted = random.randint(0, 100) <= int((100 * self.synonym_insertion_probability))
            if self.enable_random_synonym_insertion and are_synonyms_inserted:
                augmented_evidence_tokens = self._random_insertion(augmented_evidence_tokens, evidence_pos_tags, add_a_synonym=True)
                
            are_words_inserted = random.randint(0, 100) <= int((100 * self.word_insertion_probability))
            if self.enable_random_word_insertion and are_words_inserted:
                augmented_evidence_tokens = self._random_insertion(augmented_evidence_tokens, evidence_pos_tags, add_a_synonym=False)
                
            are_synonyms_deleted = random.randint(0, 100) <= int((100 * self.deletion_probability))
            if self.enable_random_deletion and are_synonyms_deleted:
                augmented_evidence_tokens = self._random_deletion(augmented_evidence_tokens)

            potential_evidence_replacements = self._process_text(
                augmented_evidence_tokens,
                evidence_pos_tags,
                claim_words=set()
            )

            num_evidence_replacements = max(0, int(len(potential_evidence_replacements) * self.replacement_fraction))
            
            if potential_evidence_replacements and num_evidence_replacements > 0:
                words_to_replace = random.sample(potential_evidence_replacements, k=num_evidence_replacements)
                current_evidence = " ".join(augmented_evidence_tokens)
                final_word_replacement_map_evidence = {}

                for word in words_to_replace:
                    lower_word = word.lower()
                    if lower_word not in evidence_pos_tags_dict or not evidence_pos_tags_dict[lower_word]:
                        continue

                    word_pos_tag = evidence_pos_tags_dict[lower_word][0]
                    synonyms = self.get_synonyms(word, word_pos_tag, topn=10)
                    if not synonyms:
                        continue

                    found, replacement = self.find_valid_replacements(
                        word,
                        synonyms,
                        current_evidence,
                        evidence_pos_tags_dict
                    )
                    if found:
                        pattern = r'\b' + re.escape(word) + r'\b'
                        try:
                            current_evidence = re.sub(pattern, replacement, current_evidence, flags=re.IGNORECASE)
                            final_word_replacement_map_evidence[word] = replacement
                            total_words_augmented += 1  # Increment the counter for each successful replacement
                        except re.error:
                            logging.warning(f"Regex error applying replacement for '{word}' with '{replacement}'.")
                augmented_evidence_text = current_evidence
            else:
                augmented_evidence_text = " ".join(augmented_evidence_tokens)

            # Remove spaces around hyphens and ensure no space after '<', '[', '(', '{'
            excluded_punctuation = "<([{"
            punctuation_to_check = ''.join(c for c in string.punctuation if c not in excluded_punctuation)
            augmented_evidence_text = re.sub(r'\s+([{}])'.format(re.escape(punctuation_to_check)), r'\1', augmented_evidence_text)
            augmented_evidence_text = re.sub(r'\s*-\s*', '-', augmented_evidence_text)  # Handle hyphens
            augmented_evidence_text = re.sub(r'([<\[({])\s+', r'\1', augmented_evidence_text)  # No space after '<', '[', '(', '{'

            # Validate the final augmented evidence text
            final_similarity_score = self.calculate_sentence_similarity(original_evidence_text, augmented_evidence_text)
            if final_similarity_score >= self.min_sentence_similarity:
                # Update the Evidence text directly in the input DataFrame
                self.train_df.at[idx, 'Evidence'] = augmented_evidence_text
                successful_augmentations += 1

        # Log final statistics
        logging.info(f"Augmentation completed. {successful_augmentations} sentences successfully augmented "
                     f"out of {attempted_augmentations} attempts.")
        logging.info(f"Total words augmented: {total_words_augmented}")
        if attempted_augmentations > 0:
            rate = (successful_augmentations / attempted_augmentations) * 100
            logging.info(f"Success rate: {rate:.2f}%")
        else:
            logging.info("No augmentation attempts were made.")

        return self.train_df


def main():
    """
    Main function to parse arguments, load data,
    run advanced synonym replacement augmentation, and save results.
    """
    parser = argparse.ArgumentParser(
        description='Advanced data augmentation using synonym replacement, random insertion, and deletion with Sentence Transformers'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=str(config.DATA_DIR / 'train_augmented_advanced_synonym_st.csv'),
        help='Output file path for augmented data'
    )
    parser.add_argument(
        '--min_sentence_similarity',
        type=float,
        default=0.85,
        help='Minimum similarity threshold between original and augmented sentences'
    )
    parser.add_argument(
        '--replacement_fraction',
        type=float,
        default=0.25,
        help='Fraction of eligible words to replace in evidence (0.0–1.0)'
    )
    parser.add_argument(
        '--st_model',
        type=str,
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='Sentence Transformer model name or path'
    )
    parser.add_argument(
        '--add_original',
        action='store_true',
        default=True,
        help='Include original evidence in the output'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help='Batch size for saving to CSV'
    )
    parser.add_argument(
        '--min_word_length',
        type=int,
        default=4,
        help='Minimum length of words to consider for replacement'
    )
    parser.add_argument(
        '--synonym_strategy',
        type=str,
        default='random',
        choices=['random', 'frequent'],
        help='Strategy for selecting synonyms: random or frequent'
    )
    parser.add_argument(
        '--allow_multi_word_synonyms',
        action='store_true',
        default=False,
        help='Allow multi-word synonyms as replacements'
    )
    parser.add_argument(
        '--word_frequency_threshold',
        type=int,
        default=5,
        help='Minimum frequency for a word to be considered for replacement'
    )
    parser.add_argument(
        '--enable_insertion',
        action='store_true',
        default=False,
        help='Enable random synonym insertion'
    )
    parser.add_argument(
        '--insertion_probability',
        type=float,
        default=0.05,
        help='Probability of inserting a synonym (per word)'
    )
    parser.add_argument(
        '--enable_deletion',
        action='store_true',
        default=False,
        help='Enable random word deletion'
    )
    parser.add_argument(
        '--deletion_probability',
        type=float,
        default=0.05,
        help='Probability of deleting a word'
    )

    args = parser.parse_args()

    output_path = config.DATA_DIR / Path(args.output_file)
    params_file = output_path.with_suffix('.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Aggregate parameters
    params = {
        'output_file': str(output_path),
        'min_sentence_similarity': args.min_sentence_similarity,
        'replacement_fraction': args.replacement_fraction,
        'sentence_transformer_model': args.st_model,
        'add_original_evidence_to_results': args.add_original,
        'batch_size': args.batch_size,
        'min_word_length': args.min_word_length,
        'synonym_selection_strategy': args.synonym_strategy,
        'allow_multi_word_synonyms': args.allow_multi_word_synonyms,
        'word_frequency_threshold': args.word_frequency_threshold,
        'enable_random_insertion': args.enable_insertion,
        'insertion_probability': args.insertion_probability,
        'enable_random_deletion': args.enable_deletion,
        'deletion_probability': args.deletion_probability,
        'date_generated': datetime.datetime.now().isoformat(),
        'synonym_source': 'WordNet'
    }

    # Save parameters to JSON
    logging.info(f"Saving parameters to {params_file}")
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=4)

    # Load data
    logging.info("Loading training data...")
    train_df = pd.read_csv(config.TRAIN_FILE)

    # Perform augmentation
    synonym_replacer = AdvancedSynonymReplacerDF(params, train_df)
    synonym_replaced_df = synonym_replacer.augment_data()

    # Save augmented data to CSV
    logging.info(f"Saving augmented data to {output_path}")
    synonym_replaced_df.to_csv(output_path, index=False)