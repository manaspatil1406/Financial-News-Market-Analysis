import re
import nltk
import spacy
from nltk.corpus import stopwords

# download once
nltk.download("stopwords")

# load spaCy model
nlp = spacy.load("en_core_web_sm")

# stopword list
stop_words = set(stopwords.words("english"))


def clean_text(text):
    # 1. lowercase
    text = str(text).lower()

    # 2. remove punctuation + numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # 3. tokenization + lemmatization
    doc = nlp(text)

    tokens = [
        token.lemma_
        for token in doc
        if token.text not in stop_words and len(token.text) > 2
    ]

    return " ".join(tokens)