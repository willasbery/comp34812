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

def load_cached_embeddings(embedding_dim=300):
    """
    Load GloVe embeddings of specified dimension from cache if available, otherwise download and cache them.
    
    Args:
        embedding_dim (int): Desired dimension for GloVe embeddings (50, 100, 200, or 300). Defaults to 300.
    
    Returns:
        dict: GloVe word embeddings dictionary.
    """
    
    # Create cache directory if it doesn't exist
    CACHE_DIR.mkdir(exist_ok=True, parents=True)
    
    cache_path = CACHE_DIR / f'glove_embeddings_{embedding_dim}.pkl'
    if cache_path.exists():
        logging.info(f"Loading GloVe embeddings from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            glove_embeddings = pickle.load(f)
    else:
        model_name = f'glove-wiki-gigaword-{embedding_dim}'
        logging.info(f"Downloading GloVe embeddings with model {model_name} (this might take a while)...")
        glove_embeddings = glove_embeddings_loader(model_name)
        
        # Cache the embeddings for future use
        logging.info(f"Caching GloVe embeddings to: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(glove_embeddings, f)
    
    return glove_embeddings

class GloveVectorizer(BaseEstimator, TransformerMixin):
    """
    A vectorizer that combines GloVe word embeddings with positional encoding.
    
    This vectorizer transforms text into fixed-size vectors by:
    1. Converting words to GloVe embeddings
    2. Applying TF-IDF weighting (optional)
    3. Adding positional encoding information
    4. Computing interaction features between claim and evidence
    """
    
    def __init__(self, sep_token: str = '[SEP]', use_tfidf_weighting=True, vocabulary=None, 
                 embedding_dim=300, ngram_range=(1,1), min_df=2, max_df=0.95):
        """
        Initialize the GloveVectorizer.
        
        Args:
            sep_token: Token used to separate claim and evidence. Defaults to '[SEP]'.
            use_tfidf_weighting: Whether to use TF-IDF weights for word embeddings. Defaults to True.
            vocabulary: Set of words to include in the vocabulary.
            embedding_dim: Desired embedding dimension.
            ngram_range: The lower and upper boundary of the n-grams to be extracted.
            min_df: Minimum document frequency for TF-IDF.
            max_df: Maximum document frequency for TF-IDF.
        """
        self.glove = load_cached_embeddings(embedding_dim)
        self.vector_size = embedding_dim
        self.sep_token = sep_token
        self.use_tfidf_weighting = use_tfidf_weighting
        self.vocabulary = vocabulary or set()
        self.ngram_range = ngram_range
        
        # Initialize TF-IDF vectorizer if weighting is enabled
        self.tfidf_vectorizer = TfidfVectorizer(
            min_df=min_df, 
            max_df=max_df,
            vocabulary=self.vocabulary,
            max_features=len(self.vocabulary) if self.vocabulary else None,
            ngram_range=self.ngram_range
        ) if use_tfidf_weighting else None
        
    @staticmethod
    def _pre_process(doc: str) -> str:
        """
        Pre-process text by removing unrepresentable characters and quotes.
        
        Args:
            doc: Input text document
            
        Returns:
            Pre-processed text with ASCII-only characters and no leading/trailing quotes
        """
        # Remove any unrepresentable characters
        doc = doc.encode('ascii', 'ignore').decode('ascii')
        # Remove any double quotes at the beginning and end of the document
        doc = doc.strip('"')
        return doc
    
    def _get_weighted_vector(self, text: str, tfidf_weights=None) -> np.ndarray:
        """
        Compute weighted average of word vectors using vocabulary words.
        
        Args:
            text: Input text
            tfidf_weights: Dictionary mapping words to their TF-IDF weights
            
        Returns:
            Weighted average vector of the input text's words
        """
        # Replace OOV words with UNK before processing
        words = [word if word in self.vocabulary else '<UNK>' for word in text.split()]
        
        if not words:
            return np.zeros(self.vector_size)
            
        # Restrict to vocabulary words
        valid_words = [word for word in words if word in self.vocabulary]
        
        if self.use_tfidf_weighting and tfidf_weights:
            vectors = []
            weights = []
            for word in valid_words:
                if word in self.glove:
                    vectors.append(self.glove[word])
                    weights.append(tfidf_weights.get(word, 1.0))  # Default weight=1 if not in TF-IDF
            
            if vectors:
                weights = np.array(weights) / np.sum(weights)  # Normalize weights
                return np.average(vectors, axis=0, weights=weights)
        
        # Fallback to regular mean if no weights or no matching words
        vectors = [self.glove[word] for word in valid_words if word in self.glove]
        if vectors:
            return np.mean(vectors, axis=0)
            
        return np.zeros(self.vector_size)
    
    def _get_positional_encoding(self, max_len: int, d_model: int) -> np.ndarray:
        """
        Generate sinusoidal positional encoding matrix.
        
        Implements the positional encoding from "Attention Is All You Need" (Vaswani et al., 2017).
        The encoding allows the model to learn to attend by relative positions.
        
        Args:
            max_len: Maximum sequence length to encode
            d_model: Dimensionality of the model/embeddings
            
        Returns:
            A matrix of shape (max_len, d_model) with positional encodings.
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
        """
        Extract features using sinusoidal positional encoding.
        
        Applies positional encoding to word vectors to capture sequential information
        in the input text.
        
        Args:
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
        """
        Fit the vectorizer by preparing TF-IDF weights if enabled.
        
        Args:
            X: Training data. Each element should be a string containing
               claim and evidence separated by self.sep_token.
            y: Target values. Not used in this vectorizer.
            
        Returns:
            self: Returns the instance itself.
        """
        if self.use_tfidf_weighting:
            # Fit TF-IDF vectorizer on all texts
            self.tfidf_vectorizer.fit([doc.replace(self.sep_token, " ") for doc in X])
        return self
    
    def transform(self, X):
        """
        Transform the input data into feature vectors.
        
        For each input text, this method:
        1. Splits the text into claim and evidence
        2. Computes weighted word embeddings for both parts
        3. Adds positional encoding information
        4. Computes interaction features between claim and evidence
        
        Args:
            X: Input data. Each element should be a string containing
               claim and evidence separated by self.sep_token.
            
        Returns:
            Feature matrix with claim, evidence, positional, and interaction features
        """
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
            
            # Prepare interaction features
            element_wise_product = claim_vector * evidence_vector
            absolute_difference = np.abs(claim_vector - evidence_vector)
            
            # Concatenate all features
            doc_vectors.append(np.concatenate([
                claim_vector, 
                evidence_vector,
                claim_pos_features,
                evidence_pos_features,
                element_wise_product,
                absolute_difference
            ]))
            
        return np.array(doc_vectors)