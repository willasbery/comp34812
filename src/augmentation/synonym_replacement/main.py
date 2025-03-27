import argparse
import logging
import nltk
import numpy as np
import pandas as pd
import pickle
import random
import re
import string
from gensim.downloader import load as glove_embeddings_loader
from nltk.corpus import stopwords as nltk_stopwords
from pathlib import Path
from tqdm import tqdm
import json
import datetime

import warnings
warnings.filterwarnings('ignore')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')

# Import config
from src.config import config

# Set up logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

# Define cache directory and path
CACHE_DIR = config.DATA_DIR.parent / "cache"
EMBEDDINGS_CACHE_PATH = CACHE_DIR / 'glove_embeddings.pkl'

stopwords = set(nltk_stopwords.words('english'))

def load_cached_embeddings():
    """Load GloVe embeddings from cache if available, otherwise download and cache them."""
    
    # Create cache directory if it doesn't exist
    CACHE_DIR.mkdir(exist_ok=True, parents=True)
    
    if EMBEDDINGS_CACHE_PATH.exists():
        logging.info(f"Loading GloVe embeddings from cache: {EMBEDDINGS_CACHE_PATH}")
        with open(EMBEDDINGS_CACHE_PATH, 'rb') as f:
            glove_embeddings = pickle.load(f)
    else:
        logging.info(f"Downloading GloVe embeddings (this might take a while)...")
        glove_embeddings = glove_embeddings_loader('glove-wiki-gigaword-300')
        
        # Cache the embeddings for future use
        logging.info(f"Caching GloVe embeddings to: {EMBEDDINGS_CACHE_PATH}")
        with open(EMBEDDINGS_CACHE_PATH, 'wb') as f:
            pickle.dump(glove_embeddings, f)
    
    return glove_embeddings

def get_synonyms(word: str, noise_level: float=0.005, topn: int=10, similarity_threshold: float=0.7):
    global glove_embeddings
    
    # Check if the word exists in the embeddings
    if word not in glove_embeddings:
        return []
    
    # Get the word's embedding vector
    original_vec = glove_embeddings[word]
    
    # Add random Gaussian noise to the vector
    noise = np.random.normal(loc=0.0, scale=noise_level, size=original_vec.shape)
    noisy_vec = original_vec + noise
    
    # Retrieve the topn + 1 words (as one of them will be the word itself) closest to the noisy vector
    # This will return a list of tuples (word, similarity)
    similar_words = glove_embeddings.most_similar(positive=[noisy_vec], topn=topn + 10)
    
    # Filter similar words by similarity threshold and return only those above threshold
    filtered_synonyms = [(syn, sim) for syn, sim in similar_words 
                         if syn != word and sim >= similarity_threshold]
    
    # Sort by similarity
    filtered_synonyms.sort(key=lambda x: x[1], reverse=True)
    
    # Return just the words from the list of tuples, limited to topn
    return [syn for syn, _ in filtered_synonyms[:topn]]


def remove_stopwords(text):
    global stopwords
    
    text = text.lower()
    
    # Remove any non-alphabetic characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Split hyphenated words
    text = re.sub(r'-', ' ', text)
    
    # Remove any double spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    
    return ' '.join([word for word in text.split() if word not in stopwords])

def process_evidence_words(claim_words: set, 
                           evidence_words: list, 
                           original_pos_tags: dict,
                           min_word_length: int = 4) -> list:
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

def find_valid_replacements(word_to_replace: str, 
                            synonyms: list, 
                            original_evidence: str, 
                            original_pos_tags: dict,
                            context_similarity_threshold: float = 0.85) -> tuple[bool, str]:
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
    context_words = [w.lower() for w in original_tokens[context_start:context_end] if w.lower() in glove_embeddings]
    if not context_words:
        return False, ""
    
    context_vecs = [glove_embeddings[w] for w in context_words]
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
        
        if pos_match:
            # Check for context similarity
            new_tokens = nltk.word_tokenize(new_evidence)
            new_word_idx = None
            
            # Find the synonym's position in the new tokens
            for i, token in enumerate(new_tokens):
                if token.lower() == synonym.lower():
                    new_word_idx = i
                    break
                    
            if new_word_idx is not None:
                # Create a context window around the synonym
                new_context_start = max(0, new_word_idx - 3)
                new_context_end = min(len(new_tokens), new_word_idx + 4)
                
                # Get the new context embedding
                new_context_words = [w.lower() for w in new_tokens[new_context_start:new_context_end] if w.lower() in glove_embeddings]
                if new_context_words:
                    new_context_vecs = [glove_embeddings[w] for w in new_context_words]
                    new_context_vec = np.mean(new_context_vecs, axis=0)
                    
                    # Calculate similarity between contexts
                    orig_norm = np.linalg.norm(original_context_vec)
                    new_norm = np.linalg.norm(new_context_vec)
                    
                    if orig_norm > 0 and new_norm > 0:
                        cosine_sim = np.dot(original_context_vec, new_context_vec) / (orig_norm * new_norm)
                        
                        if cosine_sim >= context_similarity_threshold:
                            valid_replacements.append((synonym, cosine_sim))
    
    # Sort by similarity and return the best match
    if valid_replacements:
        valid_replacements.sort(key=lambda x: x[1], reverse=True)
        return True, valid_replacements[0][0]
            
    return False, ""

def calculate_sentence_similarity(original_sentence: str, 
                                  augmented_sentence: str) -> float:
    """
    Calculate semantic similarity between original and augmented sentences.
    
    Args:
        original_sentence (str): The original sentence.
        augmented_sentence (str): The augmented sentence.
        
    Returns:
        float: Semantic similarity score between 0 and 1.
    """
    # Tokenize sentences
    original_tokens = nltk.word_tokenize(original_sentence.lower())
    augmented_tokens = nltk.word_tokenize(augmented_sentence.lower())
    
    # Get embeddings for words in both sentences
    original_vecs = [glove_embeddings[w] for w in original_tokens if w in glove_embeddings]
    augmented_vecs = [glove_embeddings[w] for w in augmented_tokens if w in glove_embeddings]
    
    if not original_vecs or not augmented_vecs:
        return 0.0
    
    # Compute sentence embeddings by averaging word vectors
    original_sentence_vec = np.mean(original_vecs, axis=0)
    augmented_sentence_vec = np.mean(augmented_vecs, axis=0)
    
    # Calculate cosine similarity
    orig_norm = np.linalg.norm(original_sentence_vec)
    aug_norm = np.linalg.norm(augmented_sentence_vec)
    
    if orig_norm > 0 and aug_norm > 0:
        cosine_sim = np.dot(original_sentence_vec, augmented_sentence_vec) / (orig_norm * aug_norm)
        return float(cosine_sim)
    
    return 0.0


def update_augment_data(train_df: pd.DataFrame, 
                 preprocessed_evidences: list, 
                 corresponding_claim: list, 
                 original_evidences: list, 
                 original_pos_tags: list, 
                 file_name: str,
                 add_original_evidence: bool = False,
                 replacement_fraction: float = 0.3,
                 min_replacement_quality: float = 0.7,
                 min_sentence_similarity: float = 0.8,
                 batch_size: int = 1000):
    """
    Enhanced function to create augmented dataset with synonym replacements and sentence similarity check.
    
    Args:
        train_df (pd.DataFrame): The training dataframe.
        preprocessed_evidences (list): List of preprocessed evidences.
        corresponding_claim (list): List of claims corresponding to the preprocessed evidences.
        original_evidences (list): List of original evidences.
        original_pos_tags (list): List of original POS tags.
        file_name (str): The file name to save the augmented data.
        add_original_evidence (bool): Whether to add the original evidence as a column.
        replacement_fraction (float): Fraction of eligible words to replace (0.0-1.0).
        min_replacement_quality (float): Minimum quality threshold for replacements.
        min_sentence_similarity (float): Minimum semantic similarity between original and augmented sentences.
        batch_size (int): The batch size for saving to CSV.
    """
    if Path(file_name).exists():
        overwrite = input(f"File {file_name} already exists. Would you like to overwrite it? (y/n) ")
        if overwrite != 'y':
            return
        
    cols = ["Claim", "Evidence", "label"]
    if add_original_evidence:
        cols.append("Original Evidence")
        cols.append("Similarity Score")
    
    synyonm_replaced_df = pd.DataFrame(columns=cols)
    batch_counter = 0
    
    successful_augmentations = 0
    attempted_augmentations = 0
    
    for idx, (claim, evidence) in tqdm(enumerate(zip(corresponding_claim, preprocessed_evidences)), 
                                     desc="Augmenting data", total=len(corresponding_claim)):
        attempted_augmentations += 1
        
        # Prepare POS tags dictionary
        pos_tags = original_pos_tags[idx]
        pos_tags_dict = {word.lower(): [] for word, _ in pos_tags}
        for word, tag in pos_tags:
            pos_tags_dict[word.lower()].append(tag)
        
        # Get potential words to replace
        claim_words = set(claim.split())
        evidence_words = evidence.split()
        potential_replacements = process_evidence_words(claim_words, evidence_words, pos_tags_dict)
        
        # Skip if not enough words to replace
        number_of_replacements = max(1, int(len(potential_replacements) * replacement_fraction))
        if number_of_replacements < 1:
            continue
        
        # Find replacements
        if len(potential_replacements) > number_of_replacements:
            words_to_replace = random.sample(potential_replacements, k=number_of_replacements)
        else:
            words_to_replace = potential_replacements
            
        final_word_replacement_map = {}
        
        for word in words_to_replace:
            synonyms = get_synonyms(word, noise_level=0.0001, topn=20, similarity_threshold=min_replacement_quality)
            found, synonym = find_valid_replacements(
                word, synonyms, original_evidences[idx], pos_tags_dict, context_similarity_threshold=0.85
            )
            if found:
                final_word_replacement_map[word] = synonym
        
        # Skip if not enough valid replacements found
        if len(final_word_replacement_map) < max(1, len(words_to_replace) * 0.5):
            continue
        
        # Create new evidence with replacements
        new_evidence = original_evidences[idx]
        for word, replacement in final_word_replacement_map.items():
            pattern = r'\b' + re.escape(word) + r'\b'
            new_evidence = re.sub(pattern, replacement, new_evidence)
        
        # Calculate semantic similarity between original and augmented sentences
        similarity_score = calculate_sentence_similarity(original_evidences[idx], new_evidence)
        
        # Skip if similarity is too low
        if similarity_score < min_sentence_similarity:
            continue
        
        # Add to dataframe
        new_row = {
            "Claim": [train_df['Claim'][idx]],
            "Evidence": [new_evidence],
            "label": [train_df['label'][idx]]
        }
        
        if add_original_evidence:
            new_row["Original Evidence"] = [original_evidences[idx]]
            new_row["Similarity Score"] = [similarity_score]
            
        new_row = pd.DataFrame(new_row)
        
        synyonm_replaced_df = pd.concat([synyonm_replaced_df, new_row], ignore_index=True)
        successful_augmentations += 1
        
        # Save batch if size threshold reached
        if len(synyonm_replaced_df) >= batch_size:
            mode = 'w' if batch_counter == 0 else 'a'
            header = batch_counter == 0
            synyonm_replaced_df.to_csv(file_name, index=False, mode=mode, header=header)
            synyonm_replaced_df = pd.DataFrame(columns=cols)
            batch_counter += 1
    
    # Save any remaining data
    if len(synyonm_replaced_df) > 0:
        mode = 'w' if batch_counter == 0 else 'a'
        header = batch_counter == 0
        
        synyonm_replaced_df.to_csv(file_name, index=False, mode=mode, header=header)
    
    # logging.info statistics
    logging.info(f"Augmentation completed: {successful_augmentations} sentences successfully augmented out of {attempted_augmentations} attempts.")
    logging.info(f"Success rate: {successful_augmentations/attempted_augmentations*100:.2f}%")


def get_diverse_replacements(word: str, 
                             context: list, 
                             topn: int=5, 
                             diversity_weight: float=0.3):
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
    global glove_embeddings
    
    # Skip if word is not in embeddings
    if word not in glove_embeddings:
        return []
    
    # Get word embedding
    word_vec = glove_embeddings[word]
    
    # Get context embeddings
    context_vecs = []
    for ctx_word in context:
        if ctx_word in glove_embeddings and ctx_word != word:
            context_vecs.append(glove_embeddings[ctx_word])
    
    if not context_vecs:
        # If no context, just return regular synonyms
        similar_words = glove_embeddings.most_similar(positive=[word_vec], topn=topn + 10)
        return [syn for syn, _ in similar_words if syn != word][:topn]
    
    # Compute context center
    context_center = np.mean(context_vecs, axis=0)
    
    # Find candidates that are similar to the target word
    candidates = glove_embeddings.most_similar(positive=[word_vec], topn=topn * 3)
    candidates = [(w, s) for w, s in candidates if w != word]
    
    # Calculate diversity scores
    diverse_candidates = []
    for candidate, similarity in candidates:
        # Skip if candidate not in embeddings
        if candidate not in glove_embeddings:
            continue
            
        candidate_vec = glove_embeddings[candidate]
        
        # Compute cosine similarity to context
        context_sim = np.dot(candidate_vec, context_center) / (
            np.linalg.norm(candidate_vec) * np.linalg.norm(context_center))
        
        # Compute diversity score (lower similarity to other candidates means more diverse)
        diversity_score = 1.0
        for other_candidate, _ in diverse_candidates:
            if other_candidate in glove_embeddings:
                other_vec = glove_embeddings[other_candidate]
                pair_sim = np.dot(candidate_vec, other_vec) / (
                    np.linalg.norm(candidate_vec) * np.linalg.norm(other_vec))
                diversity_score *= (1 - pair_sim)  # Penalize similarity to other candidates
        
        # Combined score: balance between word similarity, context fit, and diversity
        combined_score = (1 - diversity_weight) * (0.7 * similarity + 0.3 * context_sim) + diversity_weight * diversity_score
        
        diverse_candidates.append((candidate, combined_score))
        
        # Sort and limit list as we go
        diverse_candidates.sort(key=lambda x: x[1], reverse=True)
        diverse_candidates = diverse_candidates[:topn]
    
    return [w for w, _ in diverse_candidates]


def find_contextual_replacements(word_to_replace: str, 
                               original_evidence: str, 
                               original_pos_tags: dict,
                               min_similarity: float = 0.7,
                               context_window: int = 3) -> tuple[bool, str]:
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
    diverse_candidates = get_diverse_replacements(word_to_replace.lower(), context, topn=10, diversity_weight=0.3)
    
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
            similarity = calculate_sentence_similarity(original_evidence, new_evidence)
            
            if similarity >= min_similarity:
                return True, candidate
    
    return False, ""


def augment_with_chosen_method(train_df, 
                               preprocessed_evidences, 
                               corresponding_claim, 
                               original_evidences, 
                               original_pos_tags, 
                               file_name,
                               add_original_evidence, 
                               replacement_fraction, 
                               min_replacement_quality, 
                               min_sentence_similarity, 
                               batch_size,
                               use_diverse_replacements):
        # Clone the update_augment_data function but modify to use the selected replacement method
        if Path(file_name).exists():
            overwrite = input(f"File {file_name} already exists. Would you like to overwrite it? (y/n) ")
            if overwrite != 'y':
                return
        else:
            Path(file_name).parent.mkdir(parents=True, exist_ok=True)
            
        cols = ["Claim", "Evidence", "label"]
        if add_original_evidence:
            cols.append("Original Evidence")
            cols.append("Similarity Score")
        
        synyonm_replaced_df = pd.DataFrame(columns=cols)
        batch_counter = 0
        
        successful_augmentations = 0
        attempted_augmentations = 0
        
        for idx, (claim, evidence) in tqdm(enumerate(zip(corresponding_claim, preprocessed_evidences)), 
                                        desc="Augmenting data", total=len(corresponding_claim)):
            attempted_augmentations += 1
            
            # Prepare POS tags dictionary
            pos_tags = original_pos_tags[idx]
            pos_tags_dict = {word.lower(): [] for word, _ in pos_tags}
            for word, tag in pos_tags:
                pos_tags_dict[word.lower()].append(tag)
            
            # Get potential words to replace
            claim_words = set(claim.split())
            evidence_words = evidence.split()
            potential_replacements = process_evidence_words(claim_words, evidence_words, pos_tags_dict)
            
            # Skip if not enough words to replace
            number_of_replacements = max(1, int(len(potential_replacements) * replacement_fraction))
            if number_of_replacements < 1:
                continue
            
            # Find replacements
            if len(potential_replacements) > number_of_replacements:
                words_to_replace = random.sample(potential_replacements, k=number_of_replacements)
            else:
                words_to_replace = potential_replacements
                
            final_word_replacement_map = {}
            
            for word in words_to_replace:
                if use_diverse_replacements:
                    # Use contextual diverse replacements
                    found, replacement = find_contextual_replacements(
                        word, original_evidences[idx], pos_tags_dict, 
                        min_similarity=min_replacement_quality
                    )
                else:
                    # Use standard synonym replacements
                    synonyms = get_synonyms(word, noise_level=0.0001, topn=20, similarity_threshold=min_replacement_quality)
                    found, replacement = find_valid_replacements(
                        word, synonyms, original_evidences[idx], pos_tags_dict, 
                        context_similarity_threshold=0.85
                    )
                    
                if found:
                    final_word_replacement_map[word] = replacement
            
            # Skip if not enough valid replacements found
            if len(final_word_replacement_map) < max(1, len(words_to_replace) * 0.5):
                continue
            
            # Create new evidence with replacements
            new_evidence = original_evidences[idx]
            for word, replacement in final_word_replacement_map.items():
                pattern = r'\b' + re.escape(word) + r'\b'
                new_evidence = re.sub(pattern, replacement, new_evidence)
            
            # Calculate semantic similarity between original and augmented sentences
            similarity_score = calculate_sentence_similarity(original_evidences[idx], new_evidence)
            
            # Skip if similarity is too low
            if similarity_score < min_sentence_similarity:
                continue
            
            # Add to dataframe
            new_row = {
                "Claim": [train_df['Claim'][idx]],
                "Evidence": [new_evidence],
                "label": [train_df['label'][idx]]
            }
            
            if add_original_evidence:
                new_row["Original Evidence"] = [original_evidences[idx]]
                new_row["Similarity Score"] = [similarity_score]
                
            new_row = pd.DataFrame(new_row)
            
            synyonm_replaced_df = pd.concat([synyonm_replaced_df, new_row], ignore_index=True)
            successful_augmentations += 1
            
            # Save batch if size threshold reached
            if len(synyonm_replaced_df) >= batch_size:
                mode = 'w' if batch_counter == 0 else 'a'
                header = batch_counter == 0
                logging.info(f"Saving batch {batch_counter} to {file_name}")
                synyonm_replaced_df.to_csv(file_name, index=False, mode=mode, header=header)
                synyonm_replaced_df = pd.DataFrame(columns=cols)
                batch_counter += 1
        
        # Save any remaining data
        if len(synyonm_replaced_df) > 0:
            mode = 'w' if batch_counter == 0 else 'a'
            header = batch_counter == 0
            
            synyonm_replaced_df.to_csv(file_name, index=False, mode=mode, header=header)
        
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
    print(f"Saving parameters to {params_file}")
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=4)
    
    # Load embeddings
    global glove_embeddings
    glove_embeddings = load_cached_embeddings()

    logging.info(f"Loading data files...")
    train_df = pd.read_csv(config.TRAIN_FILE)
    dev_df = pd.read_csv(config.DEV_FILE)

    logging.info(f"Processing POS tags...")
    train_df['POS'] = train_df['Evidence'].apply(lambda x: nltk.pos_tag(nltk.word_tokenize(x)))
    original_evidences_pos = train_df['POS'].tolist()
    original_evidences = train_df['Evidence'].tolist()

    preprocessed_evidences = train_df['Evidence'].apply(remove_stopwords).tolist()
    corresponding_claim = train_df['Claim'].apply(remove_stopwords).tolist()

    logging.info(f"Starting data augmentation with the following parameters:")
    logging.info(f" - Minimum word similarity: {args.min_word_similarity}")
    logging.info(f" - Minimum sentence similarity: {args.min_sentence_similarity}")
    logging.info(f" - Replacement fraction: {args.replacement_fraction}")
    logging.info(f" - Using diverse contextual replacements: {args.use_diverse_replacements}")
    logging.info(f" - Output file: {args.output_file}")
    
    # Run the augmentation with chosen parameters
    num_augmented = augment_with_chosen_method(
        train_df, 
        preprocessed_evidences, 
        corresponding_claim, 
        original_evidences, 
        original_evidences_pos,
        add_original_evidence=args.add_original,
        replacement_fraction=args.replacement_fraction,
        min_replacement_quality=args.min_word_similarity,
        min_sentence_similarity=args.min_sentence_similarity,
        use_diverse_replacements=args.use_diverse_replacements,
        file_name=args.output_file,
        batch_size=args.batch_size
    )
    
    logging.info(f"Data augmentation complete. {num_augmented} new examples generated.")
    logging.info(f"New file saved at {args.output_file}")