import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import re
from scipy.spatial.distance import cosine

class FeatureExtractor:
    """A feature extractor that combines various text-based features for evidence detection.
    
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
        """Initialize the FeatureExtractor.
        
        Downloads required NLTK resources if not already present:
        - vader_lexicon: For sentiment analysis
        - punkt: For text tokenization
        """
        # Download NLTK resources if not already downloaded
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
        
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        
    def transform(self, X):
        """Transform input texts into feature vectors.
        
        For each input text (containing claim and evidence), computes:
        1. Basic text statistics:
           - Total text length
           - Claim and evidence lengths
           - Word overlap between claim and evidence
           - Word counts
           - Length ratios
           - Average word lengths
        
        2. Sentiment features:
           - Negative, neutral, positive scores for both claim and evidence
           - Compound sentiment scores
           - Absolute difference in sentiment between claim and evidence
        
        3. Text characteristics:
           - Capitalization ratios
           - Punctuation counts
           - Digit ratios
        
        4. TF-IDF similarity between claim and evidence
        
        Args:
            X (array-like): Input texts. Each element should be a string containing
                          claim and evidence separated by '[SEP]'.
            
        Returns:
            pd.DataFrame: Feature matrix with the following columns:
                - text_length: Total length of the combined text
                - claim_length: Length of the claim portion
                - evidence_length: Length of the evidence portion
                - word_overlap: Number of words that appear in both claim and evidence
                - claim_words: Number of words in claim
                - evidence_words: Number of words in evidence
                - claim_evidence_ratio: Ratio of claim length to evidence length
                - avg_word_length_claim: Average word length in claim
                - avg_word_length_evidence: Average word length in evidence
                - claim_sentiment_neg: Negative sentiment score for claim
                - claim_sentiment_neu: Neutral sentiment score for claim
                - claim_sentiment_pos: Positive sentiment score for claim
                - claim_sentiment_compound: Compound sentiment score for claim
                - evidence_sentiment_neg: Negative sentiment score for evidence
                - evidence_sentiment_neu: Neutral sentiment score for evidence
                - evidence_sentiment_pos: Positive sentiment score for evidence
                - evidence_sentiment_compound: Compound sentiment score for evidence
                - sentiment_diff: Absolute difference in compound sentiment scores
                - claim_capitals_ratio: Ratio of capital letters in claim
                - evidence_capitals_ratio: Ratio of capital letters in evidence
                - claim_punctuation_count: Number of punctuation marks in claim
                - evidence_punctuation_count: Number of punctuation marks in evidence
                - claim_digit_ratio: Ratio of digits in claim
                - evidence_digit_ratio: Ratio of digits in evidence
                - tfidf_similarity: Cosine similarity between claim and evidence TF-IDF vectors
        """
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
        """Fit the feature extractor by preparing TF-IDF weights.
        
        This method fits the TF-IDF vectorizer on all claims and evidence texts
        to prepare for computing similarity features during transform.
        
        Args:
            X (array-like): Training data. Each element should be a string containing
                          claim and evidence separated by '[SEP]'.
            y (array-like, optional): Target values. Not used in this extractor.
            
        Returns:
            self: Returns the instance itself.
        """
        # Prepare for TF-IDF
        all_texts = []
        for text in X:
            claim, evidence = text.split("[SEP]")
            all_texts.append(claim)
            all_texts.append(evidence)
        
        # Fit TF-IDF on all texts
        self.tfidf.fit(all_texts)
        return self
