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
    
    def _get_positional_encoding(self, max_len: int, d_model: int) -> np.ndarray:
        """Generate sinusoidal positional encoding matrix.
        
        Arguments:
            max_len: Maximum sequence length to encode
            d_model: Dimensionality of the model/embeddings
            
        Returns:
            A matrix of shape (max_len, d_model) with positional encodings
            
        Note:
            This implements the sinusoidal position encoding from 
            "Attention Is All You Need" (Vaswani et al., 2017).
            PE(pos, 2i) = sin(pos/10000^(2i/d_model))
            PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        """
        # Pre-compute position and dimension arrays
        positions = np.arange(max_len)[:, np.newaxis]  # Shape: (max_len, 1)
        
        # Create the div_term with proper shape for broadcasting
        div_term = np.exp(-(np.log(10000.0) / d_model) * np.arange(0, d_model, 2))
        
        # Initialize the encoding matrix
        pe = np.zeros((max_len, d_model))
        
        # Set even indices to sine, odd indices to cosine
        pe[:, 0::2] = np.sin(positions * div_term)
        if d_model > 1:  # Handle case where d_model might be 1
            pe[:, 1::2] = np.cos(positions * div_term[:pe.shape[1]//2])
            
        return pe

    def _extract_positional_features(self, text: str) -> np.ndarray:
        """Extract features using sinusoidal positional encoding.
        
        Applies positional encoding to word vectors to capture sequential information
        in the input text. This allows the model to understand the position of
        words in the sequence, similar to how transformers encode position.
        
        Arguments:
            text: Input text string
            
        Returns:
            A vector of size self.vector_size with positionally encoded features
        """
        words = text.split()
        if not words:
            return np.zeros(self.vector_size)
            
        # Get word vectors for words that exist in the embeddings
        word_vectors = []
        for word in words:
            if word in self.glove:
                word_vectors.append(self.glove[word])
        
        if not word_vectors:
            return np.zeros(self.vector_size)
            
        # Stack word vectors into a matrix: shape (sequence_length, embedding_dim)
        word_vectors = np.stack(word_vectors)
        sequence_length = word_vectors.shape[0]
        
        # Generate positional encoding of appropriate size
        pe = self._get_positional_encoding(sequence_length, self.vector_size)
        
        # Apply positional encoding to word vectors
        positionally_encoded = word_vectors + pe[:sequence_length]
        
        # Return mean of positionally encoded vectors
        return np.mean(positionally_encoded, axis=0)
    
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