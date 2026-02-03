import re
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm

# ====== Helper Functions ======
def preprocess_text(text):
    """
    Preprocess text by cleaning, lowercasing, and removing special characters.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text.strip().lower())
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

def detect_ai_content(text):
    """
    Enhanced AI content detection using linguistic patterns.
    """
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    sentence_lengths = [len(s.split()) for s in sentences]

    # Sentence Diversity Check
    if len(sentence_lengths) > 1:
        variance = np.var(sentence_lengths)
        if variance < 5:  # Low variance suggests AI content
            return "AI"

    # Vocabulary Richness Check
    words = text.split()
    unique_words = set(words)
    ttr = len(unique_words) / len(words) if words else 0
    if ttr < 0.25:  # Low TTR suggests AI content
        return "AI"

    # Repetition Analysis
    bigrams = [tuple(words[i:i + 2]) for i in range(len(words) - 1)]
    bigram_counts = Counter(bigrams)
    if any(count > 5 for count in bigram_counts.values()):  # Excessive repetition
        return "AI"

    return "Human"

def detect_spam(text, keywords):
    """
    Detect spam based on keyword density, link presence, and suspicious patterns.
    """
    words = text.split()
    keyword_density = {kw: words.count(kw) / len(words) if words else 0 for kw in keywords}

    if any(density > 0.05 for density in keyword_density.values()):  # High keyword density
        return "Spam"

    if "http" in text or "www" in text:  # Links detection
        return "Spam"

    suspicious_phrases = ["limited time offer", "act now", "winner", "free gift", "urgent response needed", "coolest", "top"]
    if any(phrase in text for phrase in suspicious_phrases):
        return "Spam"

    return "Non-Spam"

def extract_balanced_ground_truth(data, n_per_class=150):
    """
    Extract an equal number of Spam and Non-Spam URLs for ground truth.
    """
    spam_data = data[data['Spam_Label'] == "Spam"].drop_duplicates(subset='url')
    non_spam_data = data[data['Spam_Label'] == "Non-Spam"].drop_duplicates(subset='url')

    spam_sample = spam_data.sample(n=min(n_per_class, len(spam_data)), random_state=42)
    non_spam_sample = non_spam_data.sample(n=min(n_per_class, len(non_spam_data)), random_state=42)

    balanced_ground_truth = pd.concat([spam_sample, non_spam_sample]).sample(frac=1, random_state=42)  # Shuffle
    return balanced_ground_truth

# ====== Main Workflow ======
csv_file_path = "warc_extracted_5_urls.csv"
output_file = "ground_results.csv"
ground_truth_output_file = "ground_truth_results.csv"

print("Loading data...")
data = pd.read_csv(csv_file_path)

# Validate columns
required_columns = {'url', 'text', 'timestamp'}
if not required_columns.issubset(data.columns):
    raise ValueError(f"CSV file must contain columns: {required_columns}")

print("Preprocessing text...")
tqdm.pandas(desc="Processing content")
data['processed_content'] = data['text'].progress_apply(preprocess_text)

print("Detecting spam...")
spam_keywords = ["buy", "offer", "click", "deal", "discount"]
data['Spam_Label'] = data['processed_content'].apply(detect_spam, keywords=spam_keywords)

print("Detecting AI-generated content...")
data['AI_Label'] = data['processed_content'].apply(detect_ai_content)

print("Extracting balanced ground truth data...")
ground_truth = extract_balanced_ground_truth(data, n_per_class=150)
print(f"Extracted {len(ground_truth)} balanced ground truth URLs.")

print("Saving results...")
data[['url', 'timestamp', 'Spam_Label', 'AI_Label']].to_csv(output_file, index=False)
ground_truth[['url', 'timestamp', 'Spam_Label', 'AI_Label']].to_csv(ground_truth_output_file, index=False)
print(f"Results saved to {output_file}.")
print(f"Ground truth saved to {ground_truth_output_file}.")
