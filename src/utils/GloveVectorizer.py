import gensim.downloader as api
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

glove_embeddings = api.load('glove-wiki-gigaword-300')

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