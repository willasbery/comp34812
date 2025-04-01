import argparse
import datetime
import json
import logging
import nltk
import numpy as np
import pandas as pd
import random
import re
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import wordnet

import warnings
warnings.filterwarnings('ignore')

# Ensure you have these downloads available
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Import config and utility
from src.config import config
from src.utils.utils import get_device
from src.augmentation.synonym_replacement.utils import remove_stopwords

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

class SynonymReplacer:
    """
    A class to perform synonym replacement-based data augmentation using 
    NLTK WordNet and semantic similarity checks via Sentence Transformers.

    Attributes:
        params (dict): Dictionary of user-defined parameters (e.g., min sentence similarity, replacement fraction).
        train_df (pd.DataFrame): DataFrame containing 'Evidence', 'Claim', and 'label' columns.
        device (str): Torch device (e.g., 'cpu', 'cuda').
        st_model (SentenceTransformer): Loaded Sentence Transformer model for semantic similarity.
    """

    def __init__(self, params: dict, train_df: pd.DataFrame):
        """
        Initialise the SynonymReplacer with parameters and training data.

        Args:
            params (dict): Dictionary of parameters for augmentation.
            train_df (pd.DataFrame): Original training DataFrame.
        """
        self.params = params
        self.device = get_device()

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
        self.results_file_name = params.get("results_file_name", config.DATA_DIR / "synonym_replacement_results.csv")

        # Store original DataFrame and create POS tags
        self.train_df = train_df
        self.train_df['POS'] = train_df['Evidence'].apply(
            lambda x: nltk.pos_tag(nltk.word_tokenize(x))
        )

        self.original_evidences_pos = self.train_df['POS'].tolist()
        self.original_evidences = self.train_df['Evidence'].tolist()
        self.preprocessed_evidences = self.train_df['Evidence'].apply(remove_stopwords).tolist()
        self.corresponding_claim = self.train_df['Claim'].apply(remove_stopwords).tolist()

        # Log parameter summary
        logging.info("Starting data augmentation with the following parameters:")
        logging.info(f" - Sentence Transformer Model: {self.st_model_name}")
        logging.info(f" - Minimum sentence similarity: {self.min_sentence_similarity}")
        logging.info(f" - Replacement fraction: {self.replacement_fraction}")
        logging.info(f" - Output file: {self.results_file_name}")

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

    def process_evidence(
        self, 
        claim_words: set, 
        evidence_tokens: list[str], 
        pos_tags_dict: dict, 
        min_word_length: int = 4
    ) -> list[str]:
        """
        Identify potential candidate words in the evidence for replacement.

        Args:
            claim_words (set): Set of words in the claim (preprocessed).
            evidence_tokens (list[str]): Tokenised words (original form) from the evidence.
            pos_tags_dict (dict): Dictionary mapping words (lowercased) to their POS tags.
            min_word_length (int): Minimum length of words to consider for replacement.

        Returns:
            list[str]: List of tokens deemed eligible for synonym replacement.
        """
        # Identify words that overlap with the claim to avoid replacing them
        common_words = set(evidence_tokens) & claim_words
        potential_replacements = []
        # Nouns, verbs, adjectives, adverbs
        safe_pos_tags = {
            'NN', 'NNS', 'JJ', 'JJR', 'JJS', 
            'RB', 'RBR', 'RBS',
            'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'
        }

        for word in evidence_tokens:
            lower_word = word.lower()
            # Skip if:
            # 1. The word is in both claim and evidence 
            # 2. The word is a substring of any claim word (or vice versa)
            # 3. The word not in pos_tags_dict
            # 4. The word is too short
            # 5. The word's POS tags are not in the safe list
            if (word in common_words or
                any(word in cw or cw in word for cw in claim_words) or
                lower_word not in pos_tags_dict or
                len(word) < min_word_length or
                not any(tag in safe_pos_tags for tag in pos_tags_dict[lower_word])):
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

        # Map NLTK POS tags to WordNet POS tags
        wordnet_pos = None
        if pos_tag:
            if pos_tag.startswith('N'):
                wordnet_pos = wordnet.NOUN
            elif pos_tag.startswith('V'):
                wordnet_pos = wordnet.VERB
            elif pos_tag.startswith('J'):
                wordnet_pos = wordnet.ADJ
            elif pos_tag.startswith('R'):
                wordnet_pos = wordnet.ADV

        for syn in wordnet.synsets(word, pos=wordnet_pos):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower() and ' ' not in synonym:
                    synonyms.add(synonym)
                    if len(synonyms) >= topn:
                        break
            if len(synonyms) >= topn:
                break

        return list(synonyms)

    def find_valid_replacements(
        self, 
        word_to_replace: str, 
        synonyms: list[str], 
        original_evidence: str, 
        original_pos_tags: dict
    ) -> tuple[bool, str]:
        """
        From a list of synonyms, find a valid synonym replacement that maintains
        the POS tag and ensures the final augmented sentence remains semantically
        similar to the original sentence.

        Args:
            word_to_replace (str): The original word to be replaced.
            synonyms (list[str]): List of potential synonym strings.
            original_evidence (str): The unmodified evidence text.
            original_pos_tags (dict): Dictionary mapping words (lowercased) to their POS tags.

        Returns:
            tuple[bool, str]: (True, replacement_word) if a valid replacement is found, else (False, "").
        """
        valid_replacements = []
        lower_word = word_to_replace.lower()

        # Use the first POS tag of the original word as a heuristic
        original_word_pos_tags = original_pos_tags.get(lower_word, [])
        if not original_word_pos_tags:
            return False, ""
        primary_pos = original_word_pos_tags[0]

        for synonym in synonyms:
            pattern = r'\b' + re.escape(word_to_replace) + r'\b'
            try:
                new_evidence = re.sub(pattern, synonym, original_evidence, flags=re.IGNORECASE)
            except re.error:
                logging.warning(f"Regex error replacing '{word_to_replace}' with '{synonym}'. Skipping.")
                continue

            # If no replacement actually happened, skip
            if new_evidence == original_evidence:
                continue

            # Check if the synonym's POS in the new evidence matches the original POS (heuristic)
            new_evidence_pos = nltk.pos_tag(nltk.word_tokenize(new_evidence))
            synonym_pos_tags = [
                tag for (w, tag) in new_evidence_pos if w.lower() == synonym.lower()
            ]
            if not synonym_pos_tags:
                continue

            # Basic POS check
            if synonym_pos_tags[0] != primary_pos:
                continue

            # Check semantic similarity
            similarity = self.calculate_sentence_similarity(original_evidence, new_evidence)
            if similarity >= self.min_sentence_similarity:
                valid_replacements.append((synonym, similarity))

        if valid_replacements:
            # Sort and return best match by similarity
            valid_replacements.sort(key=lambda x: x[1], reverse=True)
            return True, valid_replacements[0][0]

        return False, ""

    def augment_data(self):
        """
        Perform the data augmentation on each evidence in the training DataFrame
        by replacing words with synonyms. The augmented data is saved in CSV batches.
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

        # Prepare output columns
        cols = ["Claim", "Evidence", "label"]
        if self.add_original_evidence_to_results:
            cols.append("Original Evidence")
            cols.append("Similarity Score")

        synonym_replaced_df = pd.DataFrame(columns=cols)
        batch_counter = 0

        successful_augmentations = 0
        attempted_augmentations = 0

        original_claims = self.train_df['Claim'].tolist()
        labels = self.train_df['label'].tolist()

        for idx, (claim_tokens, evidence_tokens) in tqdm(
            enumerate(zip(self.corresponding_claim, self.preprocessed_evidences)),
            desc="Augmenting data",
            total=len(self.corresponding_claim)
        ):
            attempted_augmentations += 1
            original_evidence_text = self.original_evidences[idx]

            # Build a dictionary of POS tags for the original evidence
            pos_tags = self.original_evidences_pos[idx]
            pos_tags_dict = defaultdict(list)
            for word, tag in pos_tags:
                pos_tags_dict[word.lower()].append(tag)

            # Identify potential words to replace
            claim_words_set = set(word.lower() for word in claim_tokens.split())
            original_evidence_tokens = [
                token.lower() for token in nltk.word_tokenize(original_evidence_text)
            ]
            potential_replacements = self.process_evidence(
                claim_words_set, 
                original_evidence_tokens, 
                pos_tags_dict
            )

            # Determine how many words to replace
            num_replacements = max(1, int(len(potential_replacements) * self.replacement_fraction))
            if not potential_replacements or num_replacements < 1:
                continue

            if len(potential_replacements) > num_replacements:
                words_to_replace = random.sample(potential_replacements, k=num_replacements)
            else:
                words_to_replace = potential_replacements

            final_word_replacement_map = {}
            current_evidence = original_evidence_text

            # Attempt replacements
            for word in words_to_replace:
                lower_word = word.lower()
                if lower_word not in pos_tags_dict or not pos_tags_dict[lower_word]:
                    continue

                word_pos_tag = pos_tags_dict[lower_word][0]
                synonyms = self.get_synonyms(word, word_pos_tag, topn=10)
                if not synonyms:
                    continue

                found, replacement = self.find_valid_replacements(
                    word, 
                    synonyms, 
                    current_evidence, 
                    pos_tags_dict
                )
                if found:
                    pattern = r'\b' + re.escape(word) + r'\b'
                    try:
                        current_evidence = re.sub(pattern, replacement, current_evidence, flags=re.IGNORECASE)
                        final_word_replacement_map[word] = replacement
                    except re.error:
                        logging.warning(f"Regex error applying replacement for '{word}' with '{replacement}'.")

            # Validate the final augmented text
            if len(final_word_replacement_map) < 1:
                continue

            final_similarity_score = self.calculate_sentence_similarity(original_evidence_text, current_evidence)
            if final_similarity_score < self.min_sentence_similarity:
                continue

            # Construct the new row
            new_row_data = {
                "Claim": original_claims[idx],
                "Evidence": current_evidence,
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
        if attempted_augmentations > 0:
            rate = (successful_augmentations / attempted_augmentations) * 100
            logging.info(f"Success rate: {rate:.2f}%")
        else:
            logging.info("No augmentation attempts were made.")

        return successful_augmentations


def main():
    """
    Main function to parse arguments, load data, 
    run synonym replacement augmentation, and save results.
    """
    parser = argparse.ArgumentParser(
        description='Data augmentation using synonym replacement and Sentence Transformers'
    )
    parser.add_argument(
        '--output_file', 
        type=str,
        default=str(config.DATA_DIR / 'train_augmented_synonym_st.csv'),
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
        help='Fraction of eligible words to replace (0.0–1.0)'
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
    synonym_replacer = SynonymReplacer(params, train_df)
    synonym_replacer.augment_data()


if __name__ == '__main__':
    main()
