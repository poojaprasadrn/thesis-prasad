import re
import spacy
import numpy as np
from textblob import TextBlob
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import textstat

nlp = spacy.load("en_core_web_sm")

commercial_keywords = {'buy', 'deal', 'discount', 'offer', 'sale', 'shop', 'price'}

def extract_text_features(text):
    features = {}
    if not text or not isinstance(text, str):
        return {k: 0 for k in range(15)}  # default 0s

    text = text.strip()
    doc = nlp(text)

    # --- 1. Linguistic Features ---
    token_count = len(doc)
    pos_counts = Counter([token.pos_ for token in doc])
    features['noun_ratio'] = pos_counts['NOUN'] / token_count if token_count else 0
    features['verb_ratio'] = pos_counts['VERB'] / token_count if token_count else 0
    features['adj_ratio'] = pos_counts['ADJ'] / token_count if token_count else 0
    features['adv_ratio'] = pos_counts['ADV'] / token_count if token_count else 0
    features['ner_count'] = len(list(doc.ents))
    features['avg_sent_len'] = np.mean([len(sent) for sent in doc.sents]) if len(list(doc.sents)) else 0

    # --- 2. Style & Structure ---
    words = text.split()
    features['unique_word_ratio'] = len(set(words)) / len(words) if words else 0
    features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
    features['bullet_count'] = text.count('•') + text.count('*') + text.count('- ')
    
    # --- 3. Lexical / Sentiment ---
    text_lower = text.lower()
    features['commercial_kw_count'] = sum(word in text_lower for word in commercial_keywords)
    blob = TextBlob(text)
    features['sentiment_polarity'] = blob.sentiment.polarity
    features['sentiment_subjectivity'] = blob.sentiment.subjectivity

    # --- 4. HTML-like Features (if applied to raw HTML text) ---
    features['link_density'] = text.count('http') / max(len(words), 1)

    return features
