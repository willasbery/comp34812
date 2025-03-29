import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import re
from scipy.spatial.distance import cosine

class FeatureExtractor:
    def __init__(self):
        # Download NLTK resources if not already downloaded
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
        
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        
    def transform(self, X):
        features = []
        
        for text in X:
            claim, evidence = text.split("[SEP]")
            
            # Basic features
            feature_dict = {
                'text_length': len(text),
                'claim_length': len(claim),
                'evidence_length': len(evidence),
                'word_overlap': len(set(claim.split()) & set(evidence.split())),
                'claim_words': len(claim.split()),
                'evidence_words': len(evidence.split()),
                'claim_evidence_ratio': len(claim) / max(len(evidence), 1),
                'avg_word_length_claim': np.mean([len(w) for w in claim.split()]) if claim.split() else 0,
                'avg_word_length_evidence': np.mean([len(w) for w in evidence.split()]) if evidence.split() else 0,
            }
            
            # Sentiment features
            claim_sentiment = self.sentiment_analyzer.polarity_scores(claim)
            evidence_sentiment = self.sentiment_analyzer.polarity_scores(evidence)
            
            feature_dict.update({
                'claim_sentiment_neg': claim_sentiment['neg'],
                'claim_sentiment_neu': claim_sentiment['neu'],
                'claim_sentiment_pos': claim_sentiment['pos'],
                'claim_sentiment_compound': claim_sentiment['compound'],
                'evidence_sentiment_neg': evidence_sentiment['neg'],
                'evidence_sentiment_neu': evidence_sentiment['neu'],
                'evidence_sentiment_pos': evidence_sentiment['pos'],
                'evidence_sentiment_compound': evidence_sentiment['compound'],
                'sentiment_diff': abs(claim_sentiment['compound'] - evidence_sentiment['compound'])
            })
            
            # Text characteristics
            feature_dict.update({
                'claim_capitals_ratio': sum(1 for c in claim if c.isupper()) / max(len(claim), 1),
                'evidence_capitals_ratio': sum(1 for c in evidence if c.isupper()) / max(len(evidence), 1),
                'claim_punctuation_count': sum(1 for c in claim if c in '.,;:!?'),
                'evidence_punctuation_count': sum(1 for c in evidence if c in '.,;:!?'),
                'claim_digit_ratio': sum(1 for c in claim if c.isdigit()) / max(len(claim), 1),
                'evidence_digit_ratio': sum(1 for c in evidence if c.isdigit()) / max(len(evidence), 1)
            })
            
            # TF-IDF similarity
            claim_tfidf = self.tfidf.transform([claim]).toarray()[0]
            evidence_tfidf = self.tfidf.transform([evidence]).toarray()[0]
            
            if np.sum(claim_tfidf) > 0 and np.sum(evidence_tfidf) > 0:
                tfidf_similarity = 1 - cosine(claim_tfidf, evidence_tfidf)
            else:
                tfidf_similarity = 0
                
            feature_dict['tfidf_similarity'] = tfidf_similarity
            
            features.append(feature_dict)
            
        return pd.DataFrame(features)    
    
    def fit(self, X, y=None):
        # Prepare for TF-IDF
        all_texts = []
        for text in X:
            claim, evidence = text.split("[SEP]")
            all_texts.append(claim)
            all_texts.append(evidence)
        
        # Fit TF-IDF on all texts
        self.tfidf.fit(all_texts)
        return self
