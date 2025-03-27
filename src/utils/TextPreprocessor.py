import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class TextPreprocessor:
    """
    Preprocesses the text data by:
        - converting to lowercase
        - removing special characters
        - lemmatizing the words
    """
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
    
    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text) # remove special chars
        words = word_tokenize(text)
        return ' '.join([self.lemmatizer.lemmatize(w) for w in words])
    
    def fit_transform(self, X, y=None):
        return [self.preprocess(text) for text in X]
    
    def transform(self, X):
        return [self.preprocess(text) for text in X]
    
    def fit(self, X, y=None):
        return self