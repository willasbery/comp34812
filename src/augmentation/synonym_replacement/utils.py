import re
import string
import pickle
import logging
import nltk
from gensim.downloader import load as glove_embeddings_loader
from nltk.corpus import stopwords as nltk_stopwords

from src.config import config

CACHE_DIR = config.DATA_DIR.parent / "cache"
EMBEDDINGS_CACHE_PATH = CACHE_DIR / 'glove_embeddings.pkl'

nltk.download('stopwords')
stopwords = set(nltk_stopwords.words('english'))


def load_cached_embeddings():
    """
    Load GloVe embeddings from cache if available, otherwise download and cache them.
    
    Returns:
        dict: The GloVe embeddings.
    """
    
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


def remove_stopwords(text: str) -> str:
    """
    Remove stopwords from text.
    
    Params:
        text (str): The text to remove stopwords from.
        
    Returns:
        str: The text with stopwords removed.
    """
    global stopwords
    
    text = text.lower()
    
    # Remove any non-alphabetic characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Split hyphenated words
    text = re.sub(r'-', ' ', text)
    
    # Remove any double spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    
    return ' '.join([word for word in text.split() if word not in stopwords])
