# Standard library imports
import argparse
import datetime
import json
import logging
import random
import re
from collections import defaultdict, Counter
from pathlib import Path

# Third-party imports
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import wordnet, stopwords

# Local imports
from src.config import config
from src.utils.utils import get_device
from src.augmentation.synonym_replacement.utils import remove_stopwords

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK resources
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
    Advanced data augmentation class that performs synonym replacement with semantic similarity control.
    
    This class enhances text data by replacing words with their synonyms while maintaining
    semantic meaning of the original text. It uses Sentence Transformers to verify 
    that augmented sentences remain semantically similar to the original.
    
    Attributes:
        params (dict): Configuration parameters for augmentation.
        train_df (pd.DataFrame): Training data containing 'Evidence', 'Claim', and 'label' columns.
        device (str): Computing device ('cpu' or 'cuda').
        st_model (SentenceTransformer): Model for semantic similarity measurement.
        stop_words (set): Set of English stop words to ignore during augmentation.
        word_frequencies (Counter): Word frequency counter from training data.
    """

    def __init__(self, params: dict, train_df: pd.DataFrame):
        """
        Initialize the synonym replacement augmentation with specified parameters.
        
        Args:
            params (dict): Configuration parameters for the augmentation process.
            train_df (pd.DataFrame): Training data with 'Evidence', 'Claim', and 'label' columns.
        """
        self.params = params
        self.device = get_device()
        self.stop_words = set(stopwords.words('english'))

        # Load Sentence Transformer Model
        self.st_model_name = params.get('sentence_transformer_model', 'sentence-transformers/all-MiniLM-L6-v2')
        logging.info(f"Loading Sentence Transformer model: {self.st_model_name} onto device: {self.device}")
        self.st_model = SentenceTransformer(self.st_model_name, device=self.device)
        logging.info("Sentence Transformer model loaded.")

        # Configuration parameters
        self.min_sentence_similarity = params.get("min_sentence_similarity", 0.85)
        self.replacement_fraction = params.get("replacement_fraction", 0.5)
        self.batch_size = params.get("batch_size", 1000)
        self.add_original_evidence_to_results = params.get("add_original_evidence_to_results", True)
        self.results_file_name = params.get("output_file", config.DATA_DIR / "advanced_synonym_replacement_results.csv")
        self.min_word_length = params.get("min_word_length", 4)
        self.synonym_selection_strategy = params.get("synonym_selection_strategy", "random")
        self.allow_multi_word_synonyms = params.get("allow_multi_word_synonyms", False)
        self.word_frequency_threshold = params.get("word_frequency_threshold", 5)
        
        # Advanced augmentation settings
        self.enable_random_synonym_insertion = params.get("enable_random_synonym_insertion", False)
        self.synonym_insertion_probability = params.get("synonym_insertion_probability", 0.05)
        
        self.enable_random_word_insertion = params.get("enable_random_word_insertion", False)
        self.word_insertion_probability = params.get("word_insertion_probability", 0.05)
        
        self.enable_random_deletion = params.get("enable_random_synonym_deletion", False)
        self.deletion_probability = params.get("deletion_probability", 0.05)

        # Store original DataFrame and prepare data
        self.train_df = train_df.copy()
        self._prepare_data()

        logging.info("Starting advanced data augmentation with the following parameters:")
        for key, value in params.items():
            logging.info(f" - {key}: {value}")


    def _prepare_data(self):
        """
        Prepare the training data by adding POS tags and calculating word frequencies.
        
        This method tokenizes evidence sentences, adds part-of-speech tags, and
        builds a word frequency dictionary for later use in the augmentation process.
        """
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
        Calculate the semantic similarity between two sentences.
        
        Uses the Sentence Transformer model to generate embeddings and compute
        the cosine similarity between them.
        
        Args:
            sentence_1: First sentence to compare
            sentence_2: Second sentence to compare
            
        Returns:
            float: Similarity score between 0 and 1, where 1 indicates identical meaning
        """
        embeddings = self.st_model.encode([sentence_1, sentence_2], convert_to_tensor=True, device=self.device, verbose=False)
        cosine_scores = util.cos_sim(embeddings[0], embeddings[1])
        return cosine_scores.item()


    def _get_wordnet_pos(self, tag):
        """
        Map NLTK POS tags to WordNet POS tags.
        
        Args:
            tag: NLTK part-of-speech tag
            
        Returns:
            WordNet POS constant or None if no matching tag found
        """
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
        Process text to identify candidate words eligible for replacement.
        
        Analyzes the tokens and their POS tags to determine which words can be
        replaced with synonyms based on various criteria such as word length,
        frequency, and part of speech.
        
        Args:
            text_tokens: List of tokenized words from the text
            pos_tags: List of (word, POS tag) tuples
            claim_words: Set of words in the claim (if processing evidence)
            is_claim: Whether the text being processed is a claim
            
        Returns:
            List of words eligible for synonym replacement
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

            # Skip if word is in both claim and evidence (if applicable and not processing claim)
            # or if the word is a substring of any claim word (or vice versa)
            if (not is_claim and (word in common_words or
                    any(word in cw or cw in word for cw in claim_words))):
                continue
                
            # Skip if word not in pos_tags_dict
            if lower_word not in pos_tags_dict:
                continue
                
            # Skip if word is too short
            if len(word) < self.min_word_length:
                continue
                
            # Skip if the word's POS tags are not in the safe list
            if not any(tag in safe_pos_tags for tag in pos_tags_dict[lower_word]):
                continue
                
            potential_replacements.append(word)

        return potential_replacements


    def get_synonyms(self, word: str, pos_tag: str = None, topn: int = 10) -> list[str]:
        """
        Retrieve synonyms for a word using WordNet.
        
        Finds synonyms that match the part of speech of the original word
        and applies filtering based on configuration parameters.
        
        Args:
            word: Target word to find synonyms for
            pos_tag: Part-of-speech tag of the word
            topn: Maximum number of synonyms to return
            
        Returns:
            List of potential synonyms (empty if none found)
        """
        synonyms = set()
        wordnet_pos = self._get_wordnet_pos(pos_tag) if pos_tag else None

        synsets = wordnet.synsets(word, pos=wordnet_pos)
        if not synsets:
            return []

        for syn in synsets:
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                # Only add if not the same as original word
                if synonym.lower() != word.lower():
                    # Filter multi-word synonyms based on configuration
                    if self.allow_multi_word_synonyms or ' ' not in synonym:
                        synonyms.add(synonym)
                        
                    if len(synonyms) >= topn:
                        break
                    
            if len(synonyms) >= topn:
                break

        synonym_list = list(synonyms)
        
        # Apply synonym selection strategy
        if self.synonym_selection_strategy == 'frequent':
            # Prioritize synonyms that appear more frequently in the training data
            synonym_list.sort(key=lambda s: self.word_frequencies.get(s.lower(), 0), reverse=True)
        elif self.synonym_selection_strategy == 'random':
            random.shuffle(synonym_list)

        return synonym_list[:topn]

    def get_random_word(self) -> str:
        """
        Get a random word from the training data vocabulary.
        
        Returns:
            A random word from the training data vocabulary
        """
        return random.choice(list(self.word_frequencies.keys()))


    def find_valid_replacements(
        self,
        word_to_replace: str,
        synonyms: list[str],
        original_text: str,
        original_pos_tags: dict
    ) -> tuple[bool, str]:
        """
        Find a valid synonym replacement that maintains semantic similarity.
        
        Tests each synonym to see if it:
        1. Maintains the same part of speech
        2. Keeps the sentence semantically similar to the original
        
        Args:
            word_to_replace: The original word to be replaced
            synonyms: List of potential synonym candidates
            original_text: The complete original text
            original_pos_tags: Dictionary mapping words to their POS tags
            
        Returns:
            Tuple of (success_flag, replacement_word)
        """
        lower_word = word_to_replace.lower()
        original_word_pos_tags = original_pos_tags.get(lower_word, [])
        
        if not original_word_pos_tags:
            return False, ""
            
        primary_pos = original_word_pos_tags[0]

        for synonym in synonyms:
            # Create pattern to replace only whole words (not substrings)
            pattern = r'\b' + re.escape(word_to_replace) + r'\b'
            
            try:
                new_text = re.sub(pattern, synonym, original_text, flags=re.IGNORECASE)
            except re.error:
                logging.warning(f"Regex error replacing '{word_to_replace}' with '{synonym}'. Skipping.")
                continue

            # Skip if no replacement was made
            if new_text == original_text:
                continue

            # Check if the POS tag is preserved
            new_text_pos = nltk.pos_tag(nltk.word_tokenize(new_text))
            synonym_pos_tags = [tag for (w, tag) in new_text_pos if w.lower() == synonym.lower()]
            
            if not synonym_pos_tags:
                continue

            # Verify that the WordNet POS category is maintained
            if self._get_wordnet_pos(synonym_pos_tags[0]) != self._get_wordnet_pos(primary_pos):
                continue

            # Check semantic similarity to ensure meaning is preserved
            similarity = self.calculate_sentence_similarity(original_text, new_text)
            if similarity >= self.min_sentence_similarity:
                return True, synonym

        return False, ""


    def _random_insertion(self, tokens: list[str], pos_tags: list[tuple[str, str]], add_a_synonym: bool = True) -> list[str]:
        """
        Randomly insert a word or synonym into the token list.
        
        Args:
            tokens: List of tokens to augment
            pos_tags: List of (word, POS tag) tuples
            add_a_synonym: Whether to insert a synonym (True) or random word (False)
            
        Returns:
            Augmented list of tokens with the insertion
        """
        augmented_tokens = list(tokens)
        
        if not augmented_tokens:
            return []
        
        # Choose random position and word
        insert_index = random.randint(0, len(augmented_tokens))
        word_to_augment = random.choice(augmented_tokens)
        lower_word = word_to_augment.lower()
        
        # Get POS tag for the word
        word_pos_dict = dict(pos_tags)
        original_word_pos_tag = word_pos_dict.get(lower_word)
        
        if not original_word_pos_tag:
            return augmented_tokens
        
        # Get either a synonym or random word
        if add_a_synonym:
            candidates = self.get_synonyms(word_to_augment, original_word_pos_tag, topn=5)
        else:
            candidates = [self.get_random_word()]
            
        if not candidates:
            return augmented_tokens
        
        # Insert the selected word
        word_to_insert = random.choice(candidates)
        augmented_tokens.insert(insert_index, word_to_insert)
                    
        return augmented_tokens


    def _random_deletion(self, tokens: list[str]) -> list[str]:
        """
        Randomly delete words from the token list.
        
        Deletes up to 10% of words from the input tokens.
        
        Args:
            tokens: List of tokens to augment
            
        Returns:
            Augmented list of tokens with deletions
        """
        if not tokens:
            return []
        
        # Create a copy of the tokens list
        augmented_tokens = tokens.copy()
        
        # Choose random tokens to delete (up to 10%)
        max_deletions = max(1, int(len(augmented_tokens) * 0.1))
        num_to_delete = random.randint(1, max_deletions)
        indices_to_delete = random.sample(range(len(augmented_tokens)), num_to_delete)
        
        # Create new list without the deleted tokens
        augmented_tokens = [token for i, token in enumerate(augmented_tokens) 
                           if i not in indices_to_delete]
        
        return augmented_tokens


    def augment_data(self):
        """
        Perform data augmentation on the training dataset.
        
        This method applies synonym replacement, insertion, and deletion operations
        to generate augmented versions of the evidence texts. The augmentation
        preserves the semantic similarity above the specified threshold.
        
        The augmented data is saved in batches to the specified output file.
        
        Returns:
            int: Number of successful augmentations performed
        """
        # Check if results file exists and confirm overwrite
        results_path = Path(self.results_file_name)
        if results_path.exists():
            overwrite = input(
                f"Results file {self.results_file_name} already exists. Overwrite? (y/n) "
            ).strip().lower()
            if overwrite != 'y':
                logging.info("Augmentation aborted by user.")
                return 0
        else:
            results_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare output dataframe columns
        cols = ["Claim", "Evidence", "label"]
        if self.add_original_evidence_to_results:
            cols.append("Original Evidence")
            cols.append("Similarity Score")

        # Initialize tracking variables
        synonym_replaced_df = pd.DataFrame(columns=cols)
        batch_counter = 0
        successful_augmentations = 0
        attempted_augmentations = 0
        total_words_augmented = 0

        # Get data from original dataframe
        original_claims = self.train_df['Claim'].tolist()
        labels = self.train_df['label'].tolist()

        # Process each evidence text
        for idx, original_evidence_text in tqdm(
            enumerate(self.original_evidences),
            desc="Augmenting data",
            total=len(self.original_evidences)
        ):
            attempted_augmentations += 1
            original_claim_text = original_claims[idx]

            # Get POS tagging for the evidence
            evidence_pos_tags = self.original_evidences_pos[idx]
            evidence_pos_tags_dict = defaultdict(list)
            for word, tag in evidence_pos_tags:
                evidence_pos_tags_dict[word.lower()].append(tag)
            evidence_tokens = nltk.word_tokenize(original_evidence_text)

            # Start with original tokens
            augmented_evidence_tokens = list(evidence_tokens)
            
            # Apply random augmentations based on probabilities
            # 1. Insert synonyms
            should_insert_synonyms = random.random() < self.synonym_insertion_probability
            if self.enable_random_synonym_insertion and should_insert_synonyms:
                augmented_evidence_tokens = self._random_insertion(
                    augmented_evidence_tokens, evidence_pos_tags, add_a_synonym=True)
            
            # 2. Insert random words    
            should_insert_words = random.random() < self.word_insertion_probability
            if self.enable_random_word_insertion and should_insert_words:
                augmented_evidence_tokens = self._random_insertion(
                    augmented_evidence_tokens, evidence_pos_tags, add_a_synonym=False)
            
            # 3. Delete words    
            should_delete_words = random.random() < self.deletion_probability
            if self.enable_random_deletion and should_delete_words:
                augmented_evidence_tokens = self._random_deletion(augmented_evidence_tokens)

            # Find candidate words for synonym replacement
            potential_evidence_replacements = self._process_text(
                augmented_evidence_tokens,
                evidence_pos_tags,
                claim_words=set()
            )

            # Perform the synonym replacements
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
                            total_words_augmented += 1
                        except re.error:
                            logging.warning(f"Regex error applying replacement for '{word}' with '{replacement}'.")
                
                augmented_evidence_text = current_evidence
            else:
                augmented_evidence_text = " ".join(augmented_evidence_tokens)
                
            # Clean up text formatting and spacing
            augmented_evidence_text = self._clean_text_formatting(augmented_evidence_text)

            # Validate the final augmented evidence
            final_similarity_score = self.calculate_sentence_similarity(original_evidence_text, augmented_evidence_text)
            if final_similarity_score >= self.min_sentence_similarity:
                # Add the augmented example to the results
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
                self._save_batch(synonym_replaced_df, batch_counter)
                synonym_replaced_df = pd.DataFrame(columns=cols)
                batch_counter += 1

        # Save any remaining augmented data
        if not synonym_replaced_df.empty:
            self._save_batch(synonym_replaced_df, batch_counter)

        # Log final statistics
        self._log_augmentation_stats(successful_augmentations, attempted_augmentations, total_words_augmented)

        return successful_augmentations
        
    def _clean_text_formatting(self, text: str) -> str:
        """
        Clean up text spacing and punctuation.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text with proper spacing and punctuation
        """
        # Trim whitespace
        cleaned_text = text.strip()
        
        # Ensure symbols like '$' and '£' have a space before them if not at the start
        cleaned_text = re.sub(r'(?<!\s)([$£])', r' \1', cleaned_text)
        
        # Ensure hyphens are attached with no spaces around them
        cleaned_text = re.sub(r'\s*-\s*', '-', cleaned_text)
        
        # Remove extra spaces before punctuation
        cleaned_text = re.sub(r'\s+([,.;?!])', r'\1', cleaned_text)
        
        # Adjust spacing around quotes
        cleaned_text = re.sub(r'(?<!\s)(["\"])', r' \1', cleaned_text)
        cleaned_text = re.sub(r'(["\"])(\s+)', r'\1', cleaned_text)
        
        # Collapse multiple spaces into one
        cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)
        
        return cleaned_text
        
    def _save_batch(self, df: pd.DataFrame, batch_counter: int):
        """
        Save a batch of augmented data to the output file.
        
        Args:
            df: DataFrame containing the batch data
            batch_counter: Current batch number
        """
        mode = 'w' if batch_counter == 0 else 'a'
        header = (batch_counter == 0)
        logging.info(f"Saving batch {batch_counter} to {self.results_file_name}")
        df.to_csv(self.results_file_name, index=False, mode=mode, header=header)
        
    def _log_augmentation_stats(self, successful: int, attempted: int, words_augmented: int):
        """
        Log statistics about the augmentation process.
        
        Args:
            successful: Number of successful augmentations
            attempted: Number of attempted augmentations
            words_augmented: Total number of words augmented
        """
        logging.info(f"Augmentation completed. {successful} sentences successfully augmented "
                     f"out of {attempted} attempts.")
        logging.info(f"Total words augmented: {words_augmented}")
        
        if attempted > 0:
            success_rate = (successful / attempted) * 100
            logging.info(f"Success rate: {success_rate:.2f}%")
        else:
            logging.info("No augmentation attempts were made.")


class AdvancedSynonymReplacerDF(AdvancedSynonymReplacer):
    """
    In-place synonym replacement for DataFrame augmentation.
    
    This class extends AdvancedSynonymReplacer to modify the input DataFrame directly
    without creating a separate output file. Useful for integration with transformer
    model pipelines where the entire DataFrame needs to be processed in-memory.
    
    The original evidence texts are replaced with their augmented versions
    while maintaining the same DataFrame structure.
    """
    
    def __init__(self, params: dict, train_df: pd.DataFrame):
        """
        Initialize the in-place DataFrame augmenter.
        
        Args:
            params: Configuration parameters for the augmentation process
            train_df: Training data with 'Evidence', 'Claim', and 'label' columns
        """
        super().__init__(params, train_df)
        self.train_df = train_df
        self._prepare_data()
        
    def _prepare_data(self):
        """
        Prepare the training data by adding POS tags and calculating word frequencies.
        
        Unlike the parent class, this implementation doesn't remove stopwords
        since the full text is directly modified.
        """
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
        Perform in-place data augmentation on the DataFrame.
        
        Modifies the 'Evidence' column of the input DataFrame directly with
        augmented versions of the texts.
        
        Returns:
            pd.DataFrame: Reference to the modified input DataFrame
        """
        successful_augmentations = 0
        attempted_augmentations = 0
        total_words_augmented = 0

        # Iterate through each row in the DataFrame
        for idx, row in tqdm(
            self.train_df.iterrows(),
            desc="Augmenting data",
            total=len(self.train_df)
        ):
            attempted_augmentations += 1
            original_evidence_text = row['Evidence']
            original_claim_text = row['Claim']

            # Get POS tagging for the evidence
            evidence_pos_tags = row['POS_Evidence']
            evidence_pos_tags_dict = defaultdict(list)
            for word, tag in evidence_pos_tags:
                evidence_pos_tags_dict[word.lower()].append(tag)
            evidence_tokens = nltk.word_tokenize(original_evidence_text)

            # Start with original tokens
            augmented_evidence_tokens = list(evidence_tokens)
            
            # Apply random augmentations based on probabilities
            # 1. Insert synonyms
            should_insert_synonyms = random.random() < self.synonym_insertion_probability
            if self.enable_random_synonym_insertion and should_insert_synonyms:
                augmented_evidence_tokens = self._random_insertion(
                    augmented_evidence_tokens, evidence_pos_tags, add_a_synonym=True)
            
            # 2. Insert random words    
            should_insert_words = random.random() < self.word_insertion_probability
            if self.enable_random_word_insertion and should_insert_words:
                augmented_evidence_tokens = self._random_insertion(
                    augmented_evidence_tokens, evidence_pos_tags, add_a_synonym=False)
            
            # 3. Delete words    
            should_delete_words = random.random() < self.deletion_probability
            if self.enable_random_deletion and should_delete_words:
                augmented_evidence_tokens = self._random_deletion(augmented_evidence_tokens)

            # Find candidate words for synonym replacement
            potential_evidence_replacements = self._process_text(
                augmented_evidence_tokens,
                evidence_pos_tags,
                claim_words=set()
            )

            # Perform the synonym replacements
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
                            total_words_augmented += 1
                        except re.error:
                            logging.warning(f"Regex error applying replacement for '{word}' with '{replacement}'.")
                
                augmented_evidence_text = current_evidence
            else:
                augmented_evidence_text = " ".join(augmented_evidence_tokens)

            # Clean up text formatting and spacing
            augmented_evidence_text = self._clean_text_formatting(augmented_evidence_text)

            # Validate the final augmented evidence
            final_similarity_score = self.calculate_sentence_similarity(original_evidence_text, augmented_evidence_text)
            if final_similarity_score >= self.min_sentence_similarity:
                # Update the Evidence text directly in the input DataFrame
                self.train_df.at[idx, 'Evidence'] = augmented_evidence_text
                successful_augmentations += 1

        # Log final statistics
        self._log_augmentation_stats(successful_augmentations, attempted_augmentations, total_words_augmented)

        return self.train_df


def main():
    """
    Run the advanced synonym replacement augmentation.
    
    This function parses command line arguments, configures the augmentation,
    runs it on the training data, and saves the results to the specified output file.
    """
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Prepare output paths
    output_path = config.DATA_DIR / Path(args.output_file)
    params_file = output_path.with_suffix('.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build configuration parameters
    params = construct_params_from_args(args, output_path)

    # Save parameters to JSON
    logging.info(f"Saving parameters to {params_file}")
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=4)

    # Load training data
    logging.info("Loading training data...")
    train_df = pd.read_csv(config.TRAIN_FILE)

    # Perform augmentation
    synonym_replacer = AdvancedSynonymReplacerDF(params, train_df)
    augmented_df = synonym_replacer.augment_data()

    # Save augmented data to CSV
    logging.info(f"Saving augmented data to {output_path}")
    augmented_df.to_csv(output_path, index=False)


def create_argument_parser():
    """
    Create the command-line argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='Advanced data augmentation using synonym replacement, random insertion, and deletion with Sentence Transformers'
    )
    
    # File handling
    parser.add_argument(
        '--output_file',
        type=str,
        default=str(config.DATA_DIR / 'train_augmented_advanced_synonym_st.csv'),
        help='Output file path for augmented data'
    )
    
    # Similarity and replacement settings
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
    
    # Model and output settings
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
    
    # Word selection settings
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
    
    # Advanced augmentation features
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
    
    return parser


def construct_params_from_args(args, output_path):
    """
    Convert command-line arguments to a parameters dictionary.
    
    Args:
        args: Parsed command-line arguments
        output_path: Path to the output file
        
    Returns:
        dict: Parameter dictionary for the augmenter
    """
    return {
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


if __name__ == "__main__":
    main()