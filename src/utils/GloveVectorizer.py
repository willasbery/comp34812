from gensim.downloader import load as glove_embeddings_loader
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pickle
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# Import config
from src.config import config

# Define cache directory and path
CACHE_DIR = config.DATA_DIR.parent / "cache"
EMBEDDINGS_CACHE_PATH = CACHE_DIR / 'glove_embeddings.pkl'

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

glove_embeddings = load_cached_embeddings()

class GloveVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, sep_token: str = '[SEP]', use_tfidf_weighting=True):
        self.glove = glove_embeddings
        self.vector_size = 300
        self.sep_token = sep_token
        self.use_tfidf_weighting = use_tfidf_weighting
        self.tfidf_vectorizer = TfidfVectorizer(min_df=2, max_df=0.95) if use_tfidf_weighting else None
        
    @staticmethod
    def _pre_process(doc: str) -> str:
        # Remove any unrepresentable characters
        doc = doc.encode('ascii', 'ignore').decode('ascii')
        # Remove any double quotes at the beginning and end of the document
        doc = doc.strip('"')
        return doc
    
    def _get_weighted_vector(self, text: str, tfidf_weights=None) -> np.ndarray:
        """Get weighted average of word vectors."""
        words = text.split()
        
        if not words:
            return np.zeros(self.vector_size)
            
        if self.use_tfidf_weighting and tfidf_weights:
            # Use TF-IDF weights when available
            vectors = []
            weights = []
            
            for word in words:
                if word in self.glove and word in tfidf_weights:
                    vectors.append(self.glove[word])
                    weights.append(tfidf_weights[word])
                    
            if vectors:
                weights = np.array(weights) / np.sum(weights)  # Normalize weights
                return np.average(vectors, axis=0, weights=weights)
        
        # Fallback to regular mean if no weights or no matching words
        vectors = [self.glove[word] for word in words if word in self.glove]
        if vectors:
            return np.mean(vectors, axis=0)
            
        return np.zeros(self.vector_size)
    
    def _extract_positional_features(self, text: str) -> np.ndarray:
        """Extract features based on word positions."""
        words = text.split()
        if not words:
            return np.zeros(4)
            
        # Get vectors for first and last words if they exist in embeddings
        first_word_vec = self.glove[words[0]] if words[0] in self.glove else np.zeros(self.vector_size)
        last_word_vec = self.glove[words[-1]] if words[-1] in self.glove and len(words) > 1 else np.zeros(self.vector_size)
        
        # Return mean of first and last word vectors as positional feature
        if np.any(first_word_vec) or np.any(last_word_vec):
            return np.concatenate([
                np.mean([first_word_vec], axis=0),  # First word
                np.mean([last_word_vec], axis=0),   # Last word
            ])
        return np.zeros(self.vector_size * 2)
    
    def fit(self, X, y=None):
        if self.use_tfidf_weighting:
            # Fit TF-IDF vectorizer on all texts
            self.tfidf_vectorizer.fit([doc.replace(self.sep_token, " ") for doc in X])
        return self
    
    def transform(self, X):
        doc_vectors = []
        
        # Compute TF-IDF for all documents if using weighting
        tfidf_weights_dict = {}
        if self.use_tfidf_weighting:
            # Get TF-IDF vocabulary and weights
            vocabulary = self.tfidf_vectorizer.vocabulary_
            idf = self.tfidf_vectorizer.idf_
            
            # Create a lookup dictionary for word -> tfidf weight
            tfidf_weights_dict = {word: idf[idx] for word, idx in vocabulary.items()}
        
        for doc in X:
            # Split on [SEP] token to separate claim and evidence
            try:
                claim, evidence = doc.split(self.sep_token)
            except ValueError as ve:
                raise ValueError(f"Document splitting error: Expected 2 parts separated by '{self.sep_token}', but got an error: {ve}")
            
            # Pre-process the claim and evidence
            claim = self._pre_process(claim)
            evidence = self._pre_process(evidence)
            
            # Get weighted vectors for claim and evidence
            claim_vector = self._get_weighted_vector(claim, tfidf_weights_dict)
            evidence_vector = self._get_weighted_vector(evidence, tfidf_weights_dict)
            
            # Get positional features
            claim_pos_features = self._extract_positional_features(claim)
            evidence_pos_features = self._extract_positional_features(evidence)
            
            # Concatenate all features
            doc_vectors.append(np.concatenate([
                claim_vector, 
                evidence_vector,
                claim_pos_features,
                evidence_pos_features,
                # Add interaction features
                claim_vector * evidence_vector,  # Element-wise product
                np.abs(claim_vector - evidence_vector)  # Absolute difference
            ]))
            
        return np.array(doc_vectors)