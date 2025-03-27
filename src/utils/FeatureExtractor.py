import pandas as pd

class FeatureExtractor:
    def __init__(self):
        self.feature_names = None
    
    def fit(self, X, y=None):
        # Store feature names during fit
        features = self._extract_features(X[0])  # Use first sample to get feature names
        self.feature_names = list(features.keys())
        return self
    
    def _extract_features(self, text):
        claim, evidence = text.split("[SEP]")
        return {
            'text_length': len(text),
            'claim_length': len(claim),
            'evidence_length': len(evidence),
            'word_overlap': len(set(claim.split()) & set(evidence.split())),
            'claim_words': len(claim.split()),
            'evidence_words': len(evidence.split())
        }
    
    def transform(self, X):
        features = []
        for text in X:
            features.append(self._extract_features(text))
        return pd.DataFrame(features)
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)