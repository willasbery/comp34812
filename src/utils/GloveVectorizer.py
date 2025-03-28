from gensim.downloader import load as glove_embeddings_loader
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pickle
import logging

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
    def __init__(self, sep_token: str = '[SEP]'):
        self.glove = glove_embeddings
        self.vector_size = 300
        self.sep_token = sep_token
        
    @staticmethod
    def _pre_process(doc: str) -> str:
        # Remove any unrepresentable characters
        doc = doc.encode('ascii', 'ignore').decode('ascii')
        # Remove any double quotes at the beginning and end of the document
        doc = doc.strip('"')
        return doc
    
    def _get_mean_vector(self, text: str) -> np.ndarray:
        # Get vectors for all words in text and return their mean
        vectors = [self.glove[word] for word in text.split() 
                  if word in self.glove]
        if vectors:
            return np.mean(vectors, axis=0)
        return np.zeros(self.vector_size)
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        doc_vectors = []
        
        for doc in X:
            # Split on [SEP] token to separate claim and evidence
            try:
                claim, evidence = doc.split(self.sep_token)
            except ValueError as ve:
                raise ValueError(f"Document splitting error: Expected 2 parts separated by '{self.sep_token}', but got an error: {ve}")
            
            # Pre-process the claim and evidence
            claim = self._pre_process(claim)
            evidence = self._pre_process(evidence)
            
            # Get mean vectors for claim and evidence
            claim_vector = self._get_mean_vector(claim)
            evidence_vector = self._get_mean_vector(evidence)
            
            # Concatenate claim and evidence vectors
            doc_vectors.append(np.concatenate([claim_vector, evidence_vector]))
            
        return np.array(doc_vectors)