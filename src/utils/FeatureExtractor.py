import pandas as pd

class FeatureExtractor:
    def fit_transform(self, X, y=None):
        features = []
        
        for text in X:
            claim, evidence = text.split("[SEP]")
            
            feature_dict = {
                'text_length': len(text),
                'claim_length': len(claim),
                'evidence_length': len(evidence),
                'word_overlap': len(set(claim.split()) & set(evidence.split())),
                'claim_words': len(claim.split()),
                'evidence_words': len(evidence.split())
            }
            
            features.append(feature_dict)
            
        return pd.DataFrame(features)
    
    def transform(self, X):
        return self.fit_transform(X)
    
    def fit(self, X, y=None):
        return self