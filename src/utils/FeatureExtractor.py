import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import re
from scipy.spatial.distance import cosine

class FeatureExtractor:
    """
    A feature extractor that combines various text-based features for evidence detection.
    
    This extractor computes a rich set of features including:
    1. Basic text statistics (lengths, word counts, etc.)
    2. Sentiment analysis features using VADER
    3. Text characteristics (capitalization, punctuation, digits)
    4. TF-IDF based similarity between claim and evidence
    
    The features are designed to capture both semantic and structural aspects
    of the text, which are important for evidence detection tasks.
    
    Attributes:
        sentiment_analyzer (SentimentIntensityAnalyzer): VADER sentiment analyzer
        tfidf (TfidfVectorizer): TF-IDF vectorizer for computing text similarity
    """
    
    def __init__(self):
        """
        Initialize the FeatureExtractor.
        
        Downloads required NLTK resources if not already present:
        - vader_lexicon: For sentiment analysis
        - punkt: For text tokenization
        """
        nltk.download('vader_lexicon')
        nltk.download('punkt')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        
    def transform(self, X):
        """
        Transform input texts into feature vectors.
        
        For each input text (containing claim and evidence), computes:
        1. Basic text statistics:
           - Word overlap between claim and evidence
        
        2. Sentiment features:
           - Negative, neutral, positive scores for both claim and evidence
           - Compound sentiment scores
           - Absolute difference in sentiment between claim and evidence
        
        3. TF-IDF similarity between claim and evidence
        
        Args:
            X (array-like): Input texts. Each element should be a string containing
                          claim and evidence separated by '[SEP]'.
            
        Returns:
            pd.DataFrame: Feature matrix with sentiment and similarity features.
        """
        features = []
        
        for text in X:
            claim, evidence = text.split("[SEP]")
            
            # Extract sentiment features
            claim_sentiments = self.sentiment_analyzer.polarity_scores(claim)
            evidence_sentiments = self.sentiment_analyzer.polarity_scores(evidence)
            
            # Create feature dictionary
            feature_dict = {
                'word_overlap': len(set(claim.split()) & set(evidence.split())),
                'claim_sentiment_neg': claim_sentiments['neg'],
                'claim_sentiment_neu': claim_sentiments['neu'],
                'claim_sentiment_pos': claim_sentiments['pos'],
                'claim_sentiment_compound': claim_sentiments['compound'],
                'evidence_sentiment_neg': evidence_sentiments['neg'],
                'evidence_sentiment_neu': evidence_sentiments['neu'],
                'evidence_sentiment_pos': evidence_sentiments['pos'],
                'evidence_sentiment_compound': evidence_sentiments['compound'],
                'sentiment_diff': abs(claim_sentiments['compound'] - evidence_sentiments['compound'])
            }
            
            # Calculate TF-IDF similarity
            claim_tfidf = self.tfidf.transform([claim]).toarray()[0]
            evidence_tfidf = self.tfidf.transform([evidence]).toarray()[0]
            
            # Calculate cosine similarity only if vectors are non-zero
            if np.sum(claim_tfidf) > 0 and np.sum(evidence_tfidf) > 0:
                tfidf_similarity = 1 - cosine(claim_tfidf, evidence_tfidf)
            else:
                tfidf_similarity = 0
                
            feature_dict['tfidf_similarity'] = tfidf_similarity
            features.append(feature_dict)
            
        return pd.DataFrame(features)    
    
    def fit(self, X, y=None):
        """
        Fit the feature extractor by preparing TF-IDF weights.
        
        This method fits the TF-IDF vectorizer on all claims and evidence texts
        to prepare for computing similarity features during transform.
        
        Args:
            X (array-like): Training data. Each element should be a string containing
                          claim and evidence separated by '[SEP]'.
            y (array-like, optional): Target values. Not used in this extractor.
            
        Returns:
            self: Returns the instance itself.
        """
        # Extract all texts for TF-IDF fitting
        all_texts = []
        for text in X:
            claim, evidence = text.split("[SEP]")
            all_texts.append(claim)
            all_texts.append(evidence)
        
        # Fit TF-IDF on all texts
        self.tfidf.fit(all_texts)
        return self
