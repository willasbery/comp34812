import argparse
import datetime
import json
import logging
import nltk
import numpy as np
import pandas as pd
import random
import re
from pathlib import Path
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

nltk.download('averaged_perceptron_tagger_eng')

# Import config
from src.config import config
from src.augmentation.synonym_replacement.utils import load_cached_embeddings, remove_stopwords

# Set up logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

class SynonymReplacer:
    def __init__(self, params: dict, train_df: pd.DataFrame):
        self.glove_embeddings = load_cached_embeddings()
        self.params = params
        
        self.similarity_threshold = params.get("similarity_threshold", 0.85)
        self.min_replacement_quality = params.get("min_replacement_quality", 0.7)
        self.min_sentence_similarity = params.get("min_sentence_similarity", 0.85)
        self.replacement_fraction = params.get("replacement_fraction", 0.5)
        self.batch_size = params.get("batch_size", 1000)
        self.use_diverse_replacements = params.get("use_diverse_replacements", True)
        
        self.add_original_evidence_to_results = params.get("add_original_evidence_to_results", True)
        self.results_file_name = params.get("results_file_name", "synonym_replacement_results.csv")
        
        self.train_df = train_df
        self.train_df['POS'] = train_df['Evidence'].apply(lambda x: nltk.pos_tag(nltk.word_tokenize(x)))
        self.original_evidences_pos = train_df['POS'].tolist()
        self.original_evidences = train_df['Evidence'].tolist()

        self.preprocessed_evidences = train_df['Evidence'].apply(remove_stopwords).tolist()
        self.corresponding_claim = train_df['Claim'].apply(remove_stopwords).tolist()
        
        logging.info(f"Starting data augmentation with the following parameters:")
        logging.info(f" - Minimum word similarity: {self.similarity_threshold}")
        logging.info(f" - Minimum sentence similarity: {self.min_sentence_similarity}")
        logging.info(f" - Replacement fraction: {self.replacement_fraction}")
        logging.info(f" - Using diverse contextual replacements: {self.use_diverse_replacements}")
        logging.info(f" - Output file: {self.results_file_name}")
        

    def calculate_sentence_similarity(self, sentence_1: str, sentence_2: str) -> float:
        """
        Calculate semantic similarity between two sentences.
        
        Args:
            sentence_1 (str): The first sentence.
            sentence_2 (str): The second sentence.
            
        Returns:
            float: Semantic similarity score between 0 and 1.
        """
        # Tokenize sentences
        tokens_1 = nltk.word_tokenize(sentence_1.lower())
        tokens_2 = nltk.word_tokenize(sentence_2.lower())
        
        # Get embeddings for words in both sentences
        vec_1 = [self.glove_embeddings[w] for w in tokens_1 if w in self.glove_embeddings]
        vec_2 = [self.glove_embeddings[w] for w in tokens_2 if w in self.glove_embeddings]
        
        if not vec_1 or not vec_2:
            return 0.0
        
        # Compute sentence embeddings by averaging word vectors
        sentence_1_vec = np.mean(vec_1, axis=0)
        sentence_2_vec = np.mean(vec_2, axis=0)
        
        # Calculate cosine similarity
        norm_1 = np.linalg.norm(sentence_1_vec)
        norm_2 = np.linalg.norm(sentence_2_vec)
        
        if norm_1 <= 0 or norm_2 <= 0:
            return 0.0
        
        cosine_sim = np.dot(sentence_1_vec, sentence_2_vec) / (norm_1 * norm_2)
        return float(cosine_sim)
        
        
    def process_evidence(self, claim_words: set, evidence_words: list, original_pos_tags: dict, min_word_length: int = 4) -> list:
        """
        Filter evidence words to find potential replacement candidates.
        
        Args:
            claim_words (set): Set of words in the claim.
            evidence_words (list): List of words in the evidence.
            original_pos_tags (dict): Dictionary of POS tags for the original evidence.
            min_word_length (int): Minimum length of words to consider for replacement.
            
        Returns:
            list: List of potential replacement candidates.
        """
        common_words = set(evidence_words) & claim_words
        potential_replacements = []
        
        # Define POS tags that are generally safer to replace
        safe_pos_tags = {'NN', 'NNS', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
        
        for word in evidence_words:
            # Skip if word is:
            # 1. Common between claim and evidence
            # 2. Substring of any claim word or vice versa
            # 3. Not in POS tags dictionary
            # 4. Too short (likely not meaningful enough to replace)
            # 5. Not a noun, verb, adjective or adverb (safer to replace these)
            if (word in common_words or
                any(word in claim_word or claim_word in word for claim_word in claim_words) or
                word not in original_pos_tags or
                len(word) < min_word_length or
                not any(tag in safe_pos_tags for tag in original_pos_tags[word])):
                continue
            potential_replacements.append(word)
    
        return potential_replacements
        
        
    def get_synonyms(self, word: str, noise_level: float = 0.005, topn: int = 10, similarity_threshold: float = 0.7):
        # Check if the word exists in the embeddings
        if word not in self.glove_embeddings:
            return []
        
        # Get the word's embedding vector
        original_vec = self.glove_embeddings[word]
        
        # Add random Gaussian noise to the vector
        noise = np.random.normal(loc=0.0, scale=noise_level, size=original_vec.shape)
        noisy_vec = original_vec + noise
        
        # Retrieve the topn + 1 words (as one of them will be the word itself) closest to the noisy vector
        # This will return a list of tuples (word, similarity)
        similar_words = self.glove_embeddings.most_similar(positive=[noisy_vec], topn=topn + 10)
        
        # Filter similar words by similarity threshold and return only those above threshold
        filtered_synonyms = [(syn, sim) for syn, sim in similar_words 
                            if syn != word and sim >= similarity_threshold]
        
        # Sort by similarity
        filtered_synonyms.sort(key=lambda x: x[1], reverse=True)
        
        # Return just the words from the list of tuples, limited to topn
        return [syn for syn, _ in filtered_synonyms[:topn]]
    
    
    def find_valid_replacements(self, word_to_replace: str, synonyms: list[str], original_evidence: str, original_pos_tags: dict, context_similarity_threshold: float = 0.85) -> tuple[bool, str]:
        """
        Find a valid synonym replacement that maintains POS tag and preserves meaning in context.
        
        Args:
            word_to_replace (str): The word to replace.
            synonyms (list): List of synonyms to choose from.
            original_evidence (str): The original evidence.
            original_pos_tags (dict): Dictionary of POS tags for the original evidence.
            context_similarity_threshold (float): Similarity threshold for context preservation.
            
        Returns:
            tuple[bool, str]: A tuple containing a boolean indicating if a valid replacement was found and the replacement word.
        """           
        # Create a context representation of the original sentence
        original_tokens = nltk.word_tokenize(original_evidence)
        word_idx = None
        
        # Find the word's position in the tokens
        for i, token in enumerate(original_tokens):
            if token.lower() == word_to_replace.lower():
                word_idx = i
                break
        
        if word_idx is None:
            return False, ""
        
        # Create a context window around the word
        context_start = max(0, word_idx - 3)
        context_end = min(len(original_tokens), word_idx + 4)
        
        # Get the original context embedding by averaging word vectors
        context_words = [w.lower() for w in original_tokens[context_start:context_end] if w.lower() in self.glove_embeddings]
        if not context_words:
            return False, ""
        
        context_vecs = [self.glove_embeddings[w] for w in context_words]
        original_context_vec = np.mean(context_vecs, axis=0)
        
        valid_replacements = []
        
        for synonym in synonyms:
            # Replace word in evidence
            pattern = r'\b' + re.escape(word_to_replace) + r'\b'
            new_evidence = re.sub(pattern, synonym, original_evidence)
            
            # Get POS tags for new evidence
            new_evidence_pos = nltk.pos_tag(nltk.word_tokenize(new_evidence))
            new_evidence_pos_dict = {word.lower(): [] for word, _ in new_evidence_pos}
            for word, tag in new_evidence_pos:
                new_evidence_pos_dict[word.lower()].append(tag)
            
            # Check if POS tags match
            pos_match = (word_to_replace in original_pos_tags and 
                        synonym.lower() in new_evidence_pos_dict and 
                        original_pos_tags[word_to_replace] == new_evidence_pos_dict[synonym.lower()])
            
            if not pos_match:
                continue
            
            # Check for context similarity
            new_tokens = nltk.word_tokenize(new_evidence)
            new_word_idx = None
            
            # Find the synonym's position in the new tokens
            for i, token in enumerate(new_tokens):
                if token.lower() == synonym.lower():
                    new_word_idx = i
                    break
                    
            if new_word_idx is None:
                continue
            
            # Create a context window around the synonym
            new_context_start = max(0, new_word_idx - 3)
            new_context_end = min(len(new_tokens), new_word_idx + 4)
            
            # Get the new context embedding
            new_context_words = [w.lower() for w in new_tokens[new_context_start:new_context_end] if w.lower() in self.glove_embeddings]
            if not  new_context_words:
                continue 
            
            new_context_vecs = [self.glove_embeddings[w] for w in new_context_words]
            new_context_vec = np.mean(new_context_vecs, axis=0)
            
            # Calculate similarity between contexts
            orig_norm = np.linalg.norm(original_context_vec)
            new_norm = np.linalg.norm(new_context_vec)
            
            if orig_norm <= 0 or new_norm <= 0:
                continue
            
            cosine_sim = np.dot(original_context_vec, new_context_vec) / (orig_norm * new_norm)
                
            if cosine_sim >= context_similarity_threshold:
                valid_replacements.append((synonym, cosine_sim))
        
        # Sort by similarity and return the best match
        if valid_replacements:
            valid_replacements.sort(key=lambda x: x[1], reverse=True)
            return True, valid_replacements[0][0]
                
        return False, ""


    @staticmethod
    def cosine_similarity(vec_1: list[float], vec_2: list[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        """
        return np.dot(vec_1, vec_2) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))


    def get_diverse_replacements(self, word: str, context: list[str], topn: int = 5, diversity_weight: float = 0.3):
        """
        Generate diverse but meaningful replacements for a word in context.
        
        Args:
            word (str): The word to replace.
            context (list): List of surrounding context words.
            topn (int): Number of replacements to return.
            diversity_weight (float): Weight for diversity vs. similarity (0-1).
            
        Returns:
            list: List of diverse replacement candidates.
        """        
        # Skip if word is not in embeddings
        if word not in self.glove_embeddings:
            return []
        
        # Get word embedding
        word_vec = self.glove_embeddings[word]
        
        # Get context embeddings
        context_vecs = []
        for ctx_word in context:
            if ctx_word in self.glove_embeddings and ctx_word != word:
                context_vecs.append(self.glove_embeddings[ctx_word])
        
        if not context_vecs:
            # If no context, just return regular synonyms
            similar_words = self.glove_embeddings.most_similar(positive=[word_vec], topn=topn + 10)
            return [syn for syn, _ in similar_words if syn != word][:topn]
        
        # Compute context center
        context_center = np.mean(context_vecs, axis=0)
        
        # Find candidates that are similar to the target word
        candidates = self.glove_embeddings.most_similar(positive=[word_vec], topn=topn * 3)
        candidates = [(w, s) for w, s in candidates if w != word]
        
        # Calculate diversity scores
        diverse_candidates = []
        for candidate, similarity in candidates:
            # Skip if candidate not in embeddings
            if candidate not in self.glove_embeddings:
                continue
                
            candidate_vec = self.glove_embeddings[candidate]
            
            # Compute cosine similarity to context
            context_sim = self.cosine_similarity(candidate_vec, context_center)
            
            # Compute diversity score (lower similarity to other candidates means more diverse)
            diversity_score = 1.0
            for other_candidate, _ in diverse_candidates:
                if not other_candidate in self.glove_embeddings:
                    continue
                
                other_vec = self.glove_embeddings[other_candidate]
                pair_sim = self.cosine_similarity(candidate_vec, other_vec)
                diversity_score *= (1 - pair_sim)  # Penalize similarity to other candidates
            
            # Combined score: balance between word similarity, context fit, and diversity
            combined_score = (1 - diversity_weight) * (0.7 * similarity + 0.3 * context_sim) + diversity_weight * diversity_score
            
            diverse_candidates.append((candidate, combined_score))
            diverse_candidates.sort(key=lambda x: x[1], reverse=True)
            diverse_candidates = diverse_candidates[:topn]
        
        return [w for w, _ in diverse_candidates]
    
    
    def find_contextual_replacements(self, word_to_replace: str, original_evidence: str, original_pos_tags: dict, min_similarity: float = 0.7, context_window: int = 3) -> tuple[bool, str]:
        """
        Find a contextually appropriate replacement using surrounding words.
        
        Args:
            word_to_replace (str): The word to replace.
            original_evidence (str): The original evidence.
            original_pos_tags (dict): Dictionary of POS tags for the original evidence.
            min_similarity (float): Minimum similarity threshold.
            context_window (int): Size of context window around the word.
            
        Returns:
            tuple[bool, str]: A tuple containing a boolean indicating if a valid replacement was found and the replacement word.
        """
        # Tokenize original evidence
        tokens = nltk.word_tokenize(original_evidence)
        
        # Find the word position
        word_positions = [i for i, token in enumerate(tokens) if token.lower() == word_to_replace.lower()]
        
        if not word_positions:
            return False, ""
        
        # Get context around the word
        pos = word_positions[0]  # Use first occurrence
        context_start = max(0, pos - context_window)
        context_end = min(len(tokens), pos + context_window + 1)
        context = [tokens[i].lower() for i in range(context_start, context_end) if i != pos]
        
        # Get diverse replacements
        diverse_candidates = self.get_diverse_replacements(word_to_replace.lower(), context, topn=10, diversity_weight=0.3)
        
        # Check POS tag match and semantic fit
        for candidate in diverse_candidates:
            # Replace word in evidence
            pattern = r'\b' + re.escape(word_to_replace) + r'\b'
            new_evidence = re.sub(pattern, candidate, original_evidence)
            
            # Get POS tags for new evidence
            new_evidence_pos = nltk.pos_tag(nltk.word_tokenize(new_evidence))
            new_evidence_pos_dict = {word.lower(): [] for word, _ in new_evidence_pos}
            for word, tag in new_evidence_pos:
                new_evidence_pos_dict[word.lower()].append(tag)
            
            # Check if POS tags match
            if (word_to_replace in original_pos_tags and 
                candidate.lower() in new_evidence_pos_dict and 
                original_pos_tags[word_to_replace] == new_evidence_pos_dict[candidate.lower()]):
                
                # Check sentence similarity
                similarity = self.calculate_sentence_similarity(original_evidence, new_evidence)
                
                if similarity >= min_similarity:
                    return True, candidate
        
        return False, ""

    def augment_data(self):
        if Path(self.results_file_name).exists():
            overwrite = input(f"Results file {self.results_file_name} already exists. Would you like to overwrite it? (y/n) ")
            if overwrite != 'y':
                return
        else:
            Path(self.results_file_name).parent.mkdir(parents=True, exist_ok=True)
            
        cols = ["Claim", "Evidence", "label"]
        if self.add_original_evidence_to_results:
            cols.append("Original Evidence")
            cols.append("Similarity Score")
        
        synyonm_replaced_df = pd.DataFrame(columns=cols)
        batch_counter = 0
        
        successful_augmentations = 0
        attempted_augmentations = 0
        
        for idx, (claim, evidence) in tqdm(enumerate(zip(self.corresponding_claim, self.preprocessed_evidences)), desc="Augmenting data", total=len(self.corresponding_claim)):
            attempted_augmentations += 1
            
            # Prepare POS tags dictionary
            pos_tags = self.original_evidences_pos[idx]
            pos_tags_dict = {word.lower(): [] for word, _ in pos_tags}
            for word, tag in pos_tags:
                pos_tags_dict[word.lower()].append(tag)
            
            # Get potential words to replace
            claim_words = set(claim.split())
            evidence_words = evidence.split()
            potential_replacements = self.process_evidence(claim_words, evidence_words, pos_tags_dict)
            
            # Skip if not enough words to replace
            number_of_replacements = max(1, int(len(potential_replacements) * self.replacement_fraction))
            if number_of_replacements < 1:
                continue
            
            # Find replacements, downsample if necessary
            if len(potential_replacements) > number_of_replacements:
                words_to_replace = random.sample(potential_replacements, k=number_of_replacements)
            else:
                words_to_replace = potential_replacements
                
            final_word_replacement_map = {}
            
            for word in words_to_replace:
                if self.use_diverse_replacements:
                    # Use contextual diverse replacements
                    found, replacement = self.find_contextual_replacements(
                        word, self.original_evidences[idx], pos_tags_dict, 
                        min_similarity=self.min_replacement_quality
                    )
                else:
                    # Use standard synonym replacements
                    synonyms = self.get_synonyms(word, noise_level=0.0001, topn=20, similarity_threshold=self.min_replacement_quality)
                    found, replacement = self.find_valid_replacements(
                        word, synonyms, self.original_evidences[idx], pos_tags_dict, 
                        context_similarity_threshold=0.85
                    )
                    
                if found:
                    final_word_replacement_map[word] = replacement
            
            # Skip if not enough valid replacements found
            if len(final_word_replacement_map) < max(1, len(words_to_replace) * 0.5):
                continue
            
            # Create new evidence with replacements
            new_evidence = self.original_evidences[idx]
            for word, replacement in final_word_replacement_map.items():
                pattern = r'\b' + re.escape(word) + r'\b'
                new_evidence = re.sub(pattern, replacement, new_evidence)
            
            # Calculate semantic similarity between original and augmented sentences
            similarity_score = self.calculate_sentence_similarity(self.original_evidences[idx], new_evidence)
            
            # Skip if similarity is too low
            if similarity_score < self.min_sentence_similarity:
                continue
            
            # Add to dataframe
            new_row = {
                "Claim": [self.corresponding_claim[idx]],
                "Evidence": [new_evidence],
                "label": [self.train_df['label'][idx]]
            }
            
            if self.add_original_evidence_to_results:
                new_row["Original Evidence"] = [self.original_evidences[idx]]
                new_row["Similarity Score"] = [similarity_score]
                
            new_row = pd.DataFrame(new_row)
            
            synyonm_replaced_df = pd.concat([synyonm_replaced_df, new_row], ignore_index=True)
            successful_augmentations += 1
            
            # Save batch if size threshold reached
            if len(synyonm_replaced_df) >= self.batch_size:
                mode = 'w' if batch_counter == 0 else 'a'
                header = batch_counter == 0
                logging.info(f"Saving batch {batch_counter} to {self.results_file_name}")
                synyonm_replaced_df.to_csv(self.results_file_name, index=False, mode=mode, header=header)
                synyonm_replaced_df = pd.DataFrame(columns=cols)
                batch_counter += 1
        
        # Save any remaining data
        if len(synyonm_replaced_df) > 0:
            mode = 'w' if batch_counter == 0 else 'a'
            header = batch_counter == 0
            
            synyonm_replaced_df.to_csv(self.results_file_name, index=False, mode=mode, header=header)
        
        # logging.info statistics
        logging.info(f"Augmentation completed: {successful_augmentations} sentences successfully augmented out of {attempted_augmentations} attempts.")
        logging.info(f"Success rate: {successful_augmentations / attempted_augmentations * 100:.2f}%")
        
        return successful_augmentations


def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Data augmentation with semantic preservation')
    parser.add_argument('--output_file', type=str, 
                        default=str(config.DATA_DIR / 'train_augmented_semantically_preserved.csv'),
                        help='Output file path for augmented data')
    parser.add_argument('--min_word_similarity', type=float, default=0.75,
                      help='Minimum similarity threshold for synonym replacement')
    parser.add_argument('--min_sentence_similarity', type=float, default=0.85,
                      help='Minimum similarity threshold between original and augmented sentences')
    parser.add_argument('--replacement_fraction', type=float, default=0.25,
                      help='Fraction of eligible words to replace (0.0-1.0)')
    parser.add_argument('--use_diverse_replacements', action='store_true',
                      help='Use diverse contextual replacements instead of standard synonyms')
    parser.add_argument('--add_original', action='store_true', default=True,
                      help='Include original evidence in output')
    parser.add_argument('--batch_size', type=int, default=1000,
                      help='Batch size for saving to CSV')
    
    args = parser.parse_args()
    
    # Save parameters to JSON file with same name as output file
    output_path = Path(args.output_file)
    params_file = output_path.with_suffix('.json')
    
    # Create a dictionary with all parameters
    params = {
        'output_file': args.output_file,
        'min_word_similarity': args.min_word_similarity,
        'min_sentence_similarity': args.min_sentence_similarity,
        'replacement_fraction': args.replacement_fraction,
        'use_diverse_replacements': args.use_diverse_replacements,
        'add_original': args.add_original,
        'batch_size': args.batch_size,
        'date_generated': datetime.datetime.now().isoformat(),
        'glove_model': 'glove-wiki-gigaword-300'
    }
    
    # Save parameters to JSON file
    logging.info(f"Saving parameters to {params_file}")
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=4)

    logging.info(f"Loading data files...")
    train_df = pd.read_csv(config.TRAIN_FILE)

    synonym_replacer = SynonymReplacer(params, train_df)
    synonym_replacer.augment_data()
