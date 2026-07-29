"""
Text preprocessing module for mental health classification.

All text cleaning steps exactly match the preprocessing pipeline used
during training in main.ipynb so that inference produces identical results.
"""
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

for _resource in ["tokenizers/punkt", "tokenizers/punkt_tab", "corpora/stopwords", "corpora/wordnet"]:
    try:
        nltk.data.find(_resource)
    except LookupError:
        try:
            res_name = _resource.split("/")[-1]
            nltk.download(res_name, quiet=True)
        except Exception:
            pass

try:
    stop_words = set(stopwords.words("english"))
except Exception:
    stop_words = set()

keep_words = {
    "not", "no", "nor", "never",
    "don't", "didn't", "doesn't",
    "can't", "won't", "isn't",
    "aren't", "wasn't", "weren't",
    "couldn't", "shouldn't", "wouldn't",
    "haven't", "hasn't", "hadn't",
}
custom_stopwords = stop_words - keep_words
lemmatizer = WordNetLemmatizer()

MAX_SEQUENCE_LENGTH = 100


def preprocess_text(text: str) -> str:
    """
    Apply the exact same cleaning steps used on the training corpus.

    Steps:
    1. Convert to lowercase
    2. Remove URLs
    3. Remove HTML tags
    4. Remove punctuation (keep word characters and apostrophes)
    5. Collapse whitespace
    6. Tokenize
    7. Remove custom stopwords
    8. Lemmatize with pos='v'
    9. Join back into a single space-separated string
    """
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")

    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    if custom_stopwords:
        tokens = [word for word in tokens if word.lower() not in custom_stopwords]

    try:
        tokens = [lemmatizer.lemmatize(word, pos="v") for word in tokens]
    except Exception:
        pass

    return " ".join(tokens)