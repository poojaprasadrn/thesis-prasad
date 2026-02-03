import os
import warnings
import pandas as pd
import numpy as np
import spacy
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE
import textstat
from joblib import load

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# === Load spaCy ===
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2_000_000
ai_detector = load("bert-spam-model/ai_detector_model.joblib")

# === Feature extraction function (from user) ===
def extract_features_multi(df, label_col):
    df = df.dropna(subset=["text"]).copy()
    df["text_clean"] = df["text"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True)
    tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=2000, stop_words='english')
    tfidf_feats = tfidf.fit_transform(df["text_clean"])

    pos_tags = ['VERB', 'NOUN', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'NUM']
    pos_counts = []

    for doc in nlp.pipe(df["text"].astype(str), batch_size=50):
        counts = {tag: 0 for tag in pos_tags}
        for token in doc:
            if token.pos_ in counts:
                counts[token.pos_] += 1
        pos_counts.append([counts[tag] for tag in pos_tags])

    pos_feats = np.array(pos_counts)

    df['length'] = df['text'].apply(lambda x: len(x.split()))
    df['punct'] = df['text'].apply(lambda x: sum(c in '!?.,;:' for c in x) / max(len(x.split()), 1))
    df['sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['subjectivity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    df['flesch'] = df['text'].apply(textstat.flesch_reading_ease)
    df['grade'] = df['text'].apply(textstat.flesch_kincaid_grade)
    normalized_pos = pos_feats / np.clip(df['length'].values[:, None], 1, None)
    #df['ai_prob'] = ai_detector.predict_proba(df['text'])[:, 1]  # probability of being AI-generated

    basic_feats = df[['length', 'punct', 'sentiment', 'subjectivity', 'flesch', 'grade']].values
    all_feats = np.hstack([tfidf_feats.toarray(), pos_feats, normalized_pos, basic_feats])

    selector = SelectKBest(mutual_info_classif, k=min(2500, all_feats.shape[1]))
    y = df[label_col].values
    X_selected = selector.fit_transform(all_feats, y)
    scaler = StandardScaler(with_mean=False).fit(X_selected)
    return df, X_selected, tfidf, selector, scaler

# === Filepaths ===
ground_truth_file = "rating/ground_truth_spam_with_url_final.csv"
extracted_text_file = "rating/extracted_all_urls_v2.csv"
predict_file = "extracted_new_urls_with_text.csv"

# === Load Data ===
truth = pd.read_csv(ground_truth_file)
warc = pd.read_csv(extracted_text_file)
df = pd.merge(truth, warc[['URL', 'text']], left_on="URL", right_on="URL", how="inner")

# === Create binary label: 1 = review (class-1 to class-4), 0 = other ===
df['Binary_Label'] = df['Rating'].astype(str).str.lower().isin(["class-1", "class-2", "class-3", "class-4"]).astype(int)

# === Create multiclass/review quality label: 1 = good review (class-1,2), 0 = spam review (class-3,4), np.nan otherwise ===
def review_quality(x):
    x = str(x).lower()
    if x in {"class-1", "class-2"}:
        return 1
    elif x in {"class-3", "class-4"}:
        return 0
    else:
        return np.nan
df['Review_Quality'] = df['Rating'].apply(review_quality)

# === Create spamminess score: 1 = best, 4 = spammiest ===
spam_map = {"class-1": 1, "class-2": 2, "class-3": 3, "class-4": 4}
df['Spam_Score'] = df['Rating'].astype(str).str.lower().map(spam_map)
# Only reviews (class-1..class-4) get scores, others get nan

# ================================
# === Feature Extraction      ====
# ================================

# For binary (full data, label = Binary_Label)
df_bin = df.copy()
df_bin = df_bin.dropna(subset=["text"])
df_bin['label'] = df_bin['Binary_Label']
df_bin, X_bin, tfidf_bin, selector_bin, scaler_bin = extract_features_multi(df_bin, 'label')
y_bin = df_bin['label'].values

# For multiclass (only reviews)
df_multi = df[df['Binary_Label'] == 1].copy()
df_multi = df_multi.dropna(subset=["text"])
df_multi['label'] = df_multi['Review_Quality']
df_multi = df_multi.dropna(subset=["label"])
df_multi, X_multi, tfidf_multi, selector_multi, scaler_multi = extract_features_multi(df_multi, 'label')
y_multi = df_multi['label'].values

# For spamminess regression (only reviews)
df_reg = df_multi.copy()
df_reg['label'] = df_reg['Spam_Score']
df_reg = df_reg.dropna(subset=["label"])
df_reg, X_reg, tfidf_reg, selector_reg, scaler_reg = extract_features_multi(df_reg, 'label')
y_reg = df_reg['label'].values

# =============================
# === Train/Test Split      ===
# =============================

# Binary
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X_bin, y_bin, stratify=y_bin, test_size=0.4, random_state=42)
X_train_bin, y_train_bin = SMOTE(random_state=42).fit_resample(X_train_bin, y_train_bin)
X_train_bin = scaler_bin.transform(X_train_bin)
X_test_bin = scaler_bin.transform(X_test_bin)

# Multiclass
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi, stratify=y_multi, test_size=0.4, random_state=42)
X_train_multi, y_train_multi = SMOTE(random_state=42).fit_resample(X_train_multi, y_train_multi)
X_train_multi = scaler_multi.transform(X_train_multi)
X_test_multi = scaler_multi.transform(X_test_multi)

# Regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, stratify=None, test_size=0.2, random_state=42)
X_train_reg = scaler_reg.transform(X_train_reg)
X_test_reg = scaler_reg.transform(X_test_reg)

# =============================
# === Train Models          ===
# =============================

model_bin = LogisticRegression(max_iter=700, solver='saga', class_weight='balanced')
model_bin.fit(X_train_bin, y_train_bin)

model_multi = LogisticRegression(max_iter=1000, solver='saga', class_weight='balanced')
model_multi.fit(X_train_multi, y_train_multi)

model_ridge = Ridge(max_iter=1000)
model_ridge.fit(X_train_reg, y_train_reg)

# =============================
# === Evaluation            ===
# =============================
print("=== Binary Classification ===")
y_pred_bin = model_bin.predict(X_test_bin)
print("Accuracy:", accuracy_score(y_test_bin, y_pred_bin))
print("Precision:", precision_score(y_test_bin, y_pred_bin))
print("Recall:", recall_score(y_test_bin, y_pred_bin))
print("F1 Score:", f1_score(y_test_bin, y_pred_bin))

print("\n=== Review Quality (Multiclass) ===")
y_pred_multi = model_multi.predict(X_test_multi)
print("Accuracy:", accuracy_score(y_test_multi, y_pred_multi))
print("Precision:", precision_score(y_test_multi, y_pred_multi))
print("Recall:", recall_score(y_test_multi, y_pred_multi))
print("F1 Score:", f1_score(y_test_multi, y_pred_multi))

print("\n=== Spamminess Score Regression ===")
y_pred_reg = model_ridge.predict(X_test_reg)    # <-- Add this line!
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
print("R2 Score:", r2_score(y_test_reg, y_pred_reg))

# =============================
# === Predict on New URLs   ===
# =============================

df_pred = pd.read_csv(predict_file)
df_pred = df_pred.dropna(subset=["text"])
df_pred["text_clean"] = df_pred["text"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True)

# Extract features for new URLs using the transformers and selectors fitted on the training data
tfidf_feats_pred = tfidf_bin.transform(df_pred["text_clean"])
pos_tags = ['VERB', 'NOUN', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'NUM']
pos_counts = []
for doc in nlp.pipe(df_pred["text"].astype(str), batch_size=50):
    counts = {tag: 0 for tag in pos_tags}
    for token in doc:
        if token.pos_ in counts:
            counts[token.pos_] += 1
    pos_counts.append([counts[tag] for tag in pos_tags])
pos_feats_pred = np.array(pos_counts)
df_pred['length'] = df_pred['text'].apply(lambda x: len(x.split()))
df_pred['punct'] = df_pred['text'].apply(lambda x: sum(c in '!?.,;:' for c in x) / max(len(x.split()), 1))
df_pred['sentiment'] = df_pred['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
df_pred['subjectivity'] = df_pred['text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
df_pred['flesch'] = df_pred['text'].apply(textstat.flesch_reading_ease)
df_pred['grade'] = df_pred['text'].apply(textstat.flesch_kincaid_grade)
normalized_pos_pred = pos_feats_pred / np.clip(df_pred['length'].values[:, None], 1, None)
#df['ai_prob'] = ai_detector.predict_proba(df['text'])[:, 1]
basic_feats_pred = df_pred[['length', 'punct', 'sentiment', 'subjectivity', 'flesch', 'grade']].values

all_feats_pred = np.hstack([tfidf_feats_pred.toarray(), pos_feats_pred, normalized_pos_pred, basic_feats_pred])

X_pred_bin = selector_bin.transform(scaler_bin.transform(all_feats_pred))
bin_preds = model_bin.predict(X_pred_bin)

df_pred['Binary_Class'] = np.where(bin_preds == 1, 'review', 'non-review')

# Only predict multiclass/spamminess for predicted reviews
review_mask = bin_preds == 1
X_pred_multi = selector_multi.transform(scaler_multi.transform(all_feats_pred[review_mask]))
if len(X_pred_multi) > 0:
    multi_preds = model_multi.predict(X_pred_multi)
    ridge_preds = model_ridge.predict(selector_reg.transform(scaler_reg.transform(all_feats_pred[review_mask])))
    df_pred['Multiclass_Class'] = 'others'
    df_pred['Spam_Score'] = np.nan
    df_pred.loc[review_mask, 'Multiclass_Class'] = np.where(multi_preds == 1, 'good-review', 'spam-review')
    df_pred.loc[review_mask, 'Spam_Score'] = ridge_preds
else:
    df_pred['Multiclass_Class'] = 'others'
    df_pred['Spam_Score'] = np.nan

# =============================
# === Output & Save         ===
# =============================
print("\n📊 Prediction Distribution on New URLs:")
print("Binary Classification:")
print(df_pred['Binary_Class'].value_counts(normalize=True) * 100)
print("\nMulticlass Classification:")
print(df_pred['Multiclass_Class'].value_counts(normalize=True) * 100)

df_pred[['URL', 'Binary_Class', 'Multiclass_Class', 'Spam_Score']].to_csv("predicted_new_urls_all_classes.csv", index=False)
print("\n✅ Predictions saved to predicted_new_urls_all_classes.csv")

import matplotlib.pyplot as plt

# Plot histogram of Ridge (spamminess) scores for predicted reviews
if df_pred['Spam_Score'].notna().any():
    plt.figure(figsize=(6, 4))
    plt.hist(df_pred['Spam_Score'].dropna(), bins=10, edgecolor='k')
    plt.title("Predicted Spamminess Score (Ridge Regression)\n[For predicted reviews]")
    plt.xlabel("Predicted Spamminess Score")
    plt.ylabel("Count")
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig("predicted_spamminess_hist.png", dpi=300) 
    plt.show()
else:
    print("No predicted reviews to plot spamminess score.")

# Regression scatterplot for test set
plt.figure(figsize=(6,4))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.7)
plt.title("Ridge Regression: Ground Truth vs Predicted Spam Score (Test Set)")
plt.xlabel("True Spamminess Score")
plt.ylabel("Predicted Spamminess Score")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("ridge_regression_scatter_test.png", dpi=300)
plt.show()


# # === FULL FINAL CODE WITH EVERYTHING INCLUDED ===

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import warnings
# warnings.filterwarnings('ignore')
# import pandas as pd
# import numpy as np
# import re
# import spacy
# from textblob import TextBlob
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_selection import SelectKBest, mutual_info_classif
# from imblearn.over_sampling import SMOTE
# from scipy.sparse import hstack

# # === Load spaCy model ===
# nlp = spacy.load("en_core_web_sm")
# nlp.max_length = 2_000_000

# # === Feature Functions ===
# def preprocess_text(text):
#     if pd.isna(text) or not isinstance(text, str):
#         return ""
#     text = text.strip().lower()
#     text = re.sub(r'\s+', ' ', text)
#     text = re.sub(r'[^\w\s]', '', text)
#     return text

# def extract_pos_tags(text):
#     doc = nlp(text)
#     return ' '.join([token.pos_ for token in doc])

# def extract_sentiment(text):
#     return TextBlob(text).sentiment.polarity

# def extract_subjectivity(text):
#     return TextBlob(text).sentiment.subjectivity

# def readability_features(df):
#     sentence_length = np.array([[len(text.split()) for text in df['text']]]).T
#     punctuation_ratio = np.array([[sum(1 for char in text if char in '!?.,;:') / max(len(text.split()), 1) for text in df['text']]]).T
#     avg_word_length = np.array([[np.mean([len(word) for word in text.split()]) if text.split() else 0 for text in df['text']]]).T
#     return sentence_length, punctuation_ratio, avg_word_length

# def add_pattern_features(df):
#     patterns = [
#         r'we tested', r'we reviewed', r'i think', r'i feel', r'my pick',
#         r'compared to', r'vs', r'our verdict', r'in conclusion', r'out of 5',
#         r'\brating\b', r'best [a-z]+', r'top \d+', r'pros and cons', r'unboxing'
#     ]
#     for i, pat in enumerate(patterns):
#         df[f'pattern_{i}'] = df['text'].str.contains(pat, case=False, regex=True).astype(int)
#     return df

# def add_url_features(df):
#     df['has_review_url'] = df['URL'].str.contains(r'review|compare|vs|test|top|best|buy|product', case=False).astype(int)
#     return df

# # === Load and Preprocess Data ===
# ground_truth = pd.read_csv("rating/ground_truth_spam_with_url_final.csv")
# warc_data = pd.read_csv("rating/extracted_all_urls.csv")
# merged_data = pd.merge(ground_truth, warc_data[['URL', 'text']], on='URL', how='left')
# merged_data['text'] = merged_data['text'].apply(preprocess_text)
# merged_data['POS'] = merged_data['text'].apply(extract_pos_tags)
# merged_data = add_pattern_features(merged_data)
# merged_data = add_url_features(merged_data)
# merged_data['label'] = merged_data['Rating'].apply(lambda x: 1 if str(x).strip().lower() in {"class-1", "class-2", "class-3", "class-4"} else 0)

# # === Feature Extraction ===
# tfidf_vec = TfidfVectorizer(max_features=1500, stop_words='english', ngram_range=(1, 2))
# X_text = tfidf_vec.fit_transform(merged_data['text'])
# pos_vec = CountVectorizer()
# X_pos = pos_vec.fit_transform(merged_data['POS'])
# sentiment = np.array([[extract_sentiment(text) for text in merged_data['text']]]).T
# subjectivity = np.array([[extract_subjectivity(text) for text in merged_data['text']]]).T
# sentence_length, punctuation_ratio, avg_word_length = readability_features(merged_data)
# X_pattern = merged_data[[col for col in merged_data.columns if col.startswith('pattern_')]].values
# X_url = merged_data[['has_review_url']].values
# X_all = hstack([X_text, X_pos, sentiment, subjectivity, sentence_length, punctuation_ratio, avg_word_length, X_pattern, X_url])

# # === Binary Classification ===
# y_bin = merged_data['label'].values
# X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X_all, y_bin, stratify=y_bin, test_size=0.3, random_state=42)
# X_train_bin, y_train_bin = SMOTE(random_state=42).fit_resample(X_train_bin, y_train_bin)
# scaler = StandardScaler(with_mean=False)
# X_train_bin = scaler.fit_transform(X_train_bin)
# X_test_bin = scaler.transform(X_test_bin)
# selector = SelectKBest(score_func=mutual_info_classif, k=min(1500, X_train_bin.shape[1]))
# X_train_bin = selector.fit_transform(X_train_bin, y_train_bin)
# X_test_bin = selector.transform(X_test_bin)
# model_bin = LogisticRegression(max_iter=500, solver='saga', class_weight='balanced')
# model_bin.fit(X_train_bin, y_train_bin)
# y_pred_bin = model_bin.predict(X_test_bin)

# print("\n📊 Binary Classification (Review vs Non-Review)")
# print(f"Accuracy: {accuracy_score(y_test_bin, y_pred_bin):.2f}")
# print(f"Precision: {precision_score(y_test_bin, y_pred_bin):.2f}")
# print(f"Recall: {recall_score(y_test_bin, y_pred_bin):.2f}")
# print(f"F1 Score: {f1_score(y_test_bin, y_pred_bin):.2f}")

# # Predict binary for full dataset
# merged_data['Binary_Pred'] = model_bin.predict(selector.transform(scaler.transform(X_all)))
# merged_data['Binary_Class'] = merged_data['Binary_Pred'].map({1: 'review', 0: 'non-review'})

# # === Multiclass Classification using model_r only on predicted reviews ===
# def map_review_quality(x):
#     x = str(x).strip().lower()
#     if x in {'class-1', 'class-2'}:
#         return 'high-quality-review'
#     elif x in {'class-3', 'class-4'}:
#         return 'low-quality-review'
#     return np.nan

# review_data = merged_data[merged_data['label'] == 1].copy()
# review_data['Review_Quality'] = review_data['Rating'].apply(map_review_quality)
# review_data = review_data.dropna(subset=['Review_Quality'])

# X_text_r = tfidf_vec.transform(review_data['text'])
# X_pos_r = pos_vec.transform(review_data['POS'])
# sentiment_r = np.array([[extract_sentiment(text) for text in review_data['text']]]).T
# subjectivity_r = np.array([[extract_subjectivity(text) for text in review_data['text']]]).T
# sl_r, pr_r, wl_r = readability_features(review_data)
# X_pattern_r = review_data[[col for col in review_data.columns if col.startswith('pattern_')]].values
# X_url_r = review_data[['has_review_url']].values
# X_review = hstack([X_text_r, X_pos_r, sentiment_r, subjectivity_r, sl_r, pr_r, wl_r, X_pattern_r, X_url_r])
# y_review = (review_data['Review_Quality'] == 'high-quality-review').astype(int)

# X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_review, y_review, stratify=y_review, test_size=0.3, random_state=42)
# X_train_r, y_train_r = SMOTE(random_state=42).fit_resample(X_train_r, y_train_r)
# X_train_r = scaler.fit_transform(X_train_r)
# X_test_r = scaler.transform(X_test_r)
# X_train_r = selector.fit_transform(X_train_r, y_train_r)
# X_test_r = selector.transform(X_test_r)
# model_r = LogisticRegression(max_iter=1000, solver='saga', class_weight='balanced')
# model_r.fit(X_train_r, y_train_r)
# y_pred_r = model_r.predict(X_test_r)

# print("\n📊 Review Subset: High vs Low Quality")
# print(f"Accuracy: {accuracy_score(y_test_r, y_pred_r):.2f}")
# print(f"Precision: {precision_score(y_test_r, y_pred_r):.2f}")
# print(f"Recall: {recall_score(y_test_r, y_pred_r):.2f}")
# print(f"F1 Score: {f1_score(y_test_r, y_pred_r):.2f}")

# # Predict Multiclass for predicted reviews
# review_mask = merged_data['Binary_Pred'] == 1
# X_text_all = tfidf_vec.transform(merged_data.loc[review_mask, 'text'])
# X_pos_all = pos_vec.transform(merged_data.loc[review_mask, 'POS'])
# sentiment_all = np.array([[extract_sentiment(t) for t in merged_data.loc[review_mask, 'text']]]).T
# subjectivity_all = np.array([[extract_subjectivity(t) for t in merged_data.loc[review_mask, 'text']]]).T
# sl_all, pr_all, wl_all = readability_features(merged_data.loc[review_mask])
# X_pattern_all = merged_data.loc[review_mask, [col for col in merged_data.columns if col.startswith('pattern_')]].values
# X_url_all = merged_data.loc[review_mask, ['has_review_url']].values

# X_review_all = hstack([X_text_all, X_pos_all, sentiment_all, subjectivity_all, sl_all, pr_all, wl_all, X_pattern_all, X_url_all])
# X_review_all = scaler.transform(X_review_all)
# X_review_all = selector.transform(X_review_all)

# review_preds = model_r.predict(X_review_all)
# multiclass = np.array(['others'] * len(merged_data))
# multiclass[review_mask] = np.where(review_preds == 1, 'good-review', 'spam-review')
# merged_data['Multiclass_Class'] = multiclass

# # Save final predictions
# merged_data[['URL', 'Binary_Class', 'Multiclass_Class']].to_csv("final_review_predictions.csv", index=False)
# print("\n✅ Final model-based predictions saved to final_review_predictions.csv")

# # Print prediction distribution
# bin_dist = merged_data['Binary_Class'].value_counts(normalize=True) * 100
# multi_dist = merged_data['Multiclass_Class'].value_counts(normalize=True) * 100
# print("\n📊 Prediction Distribution")
# print("Binary Classification:")
# print(bin_dist.round(2).to_string())
# print("\nMulticlass Classification:")
# print(multi_dist.round(2).to_string())


# # === Imports ===

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import warnings
# warnings.filterwarnings('ignore')
# import pandas as pd
# import numpy as np
# import re
# import spacy
# from textblob import TextBlob
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_selection import SelectKBest, mutual_info_classif
# from imblearn.over_sampling import SMOTE
# from collections import Counter
# from scipy.sparse import hstack

# # === Load spaCy model ===
# nlp = spacy.load("en_core_web_sm")
# nlp.max_length = 2_000_000

# # === Feature Functions ===
# def preprocess_text(text):
#     if pd.isna(text) or not isinstance(text, str):
#         return ""
#     text = text.strip().lower()
#     text = re.sub(r'\s+', ' ', text)
#     text = re.sub(r'[^\w\s]', '', text)
#     return text

# def extract_pos_tags(text):
#     doc = nlp(text)
#     return ' '.join([token.pos_ for token in doc])

# def extract_sentiment(text):
#     return TextBlob(text).sentiment.polarity

# def extract_subjectivity(text):
#     return TextBlob(text).sentiment.subjectivity

# def readability_features(df):
#     sentence_length = np.array([[len(text.split()) for text in df['text']]]).T
#     punctuation_ratio = np.array([[
#         sum(1 for char in text if char in '!?.,;:') / max(len(text.split()), 1)
#         for text in df['text']
#     ]]).T
#     avg_word_length = np.array([[np.mean([len(word) for word in text.split()]) if text.split() else 0 for text in df['text']]]).T
#     return sentence_length, punctuation_ratio, avg_word_length

# def add_pattern_features(df):
#     patterns = [
#         r'we tested', r'we reviewed', r'i think', r'i feel', r'my pick',
#         r'compared to', r'vs', r'our verdict', r'in conclusion', r'out of 5',
#         r'\brating\b', r'best [a-z]+', r'top \d+', r'pros and cons', r'unboxing'
#     ]
#     for i, pat in enumerate(patterns):
#         df[f'pattern_{i}'] = df['text'].str.contains(pat, case=False, regex=True).astype(int)
#     return df

# def add_url_features(df):
#     df['has_review_url'] = df['URL'].str.contains(r'review|compare|vs|test|top|best|buy|product', case=False).astype(int)
#     return df

# # === Load and Preprocess Data ===
# ground_truth = pd.read_csv("rating/ground_truth_spam_with_url_final.csv")
# warc_data = pd.read_csv("rating/extracted_all_urls.csv")
# merged_data = pd.merge(ground_truth, warc_data[['URL', 'text']], on='URL', how='left')
# merged_data['text'] = merged_data['text'].apply(preprocess_text)
# merged_data['POS'] = merged_data['text'].apply(extract_pos_tags)
# merged_data = add_pattern_features(merged_data)
# merged_data = add_url_features(merged_data)
# merged_data['label'] = merged_data['Rating'].apply(lambda x: 1 if str(x).strip().lower() in {"class-1", "class-2", "class-3", "class-4"} else 0)

# # === Feature Extraction ===
# tfidf_vec = TfidfVectorizer(max_features=1500, stop_words='english', ngram_range=(1, 2))
# X_text = tfidf_vec.fit_transform(merged_data['text'])
# pos_vec = CountVectorizer()
# X_pos = pos_vec.fit_transform(merged_data['POS'])
# sentiment = np.array([[extract_sentiment(text) for text in merged_data['text']]]).T
# subjectivity = np.array([[extract_subjectivity(text) for text in merged_data['text']]]).T
# sentence_length, punctuation_ratio, avg_word_length = readability_features(merged_data)
# X_pattern = merged_data[[col for col in merged_data.columns if col.startswith('pattern_')]].values
# X_url = merged_data[['has_review_url']].values
# X_all = hstack([X_text, X_pos, sentiment, subjectivity, sentence_length, punctuation_ratio, avg_word_length, X_pattern, X_url])

# # === Binary Classification ===
# y = merged_data['label'].values
# X_train, X_test, y_train, y_test = train_test_split(X_all, y, stratify=y, test_size=0.3, random_state=42)
# X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
# scaler = StandardScaler(with_mean=False)
# X_train_scaled = scaler.fit_transform(X_train_res)
# X_test_scaled = scaler.transform(X_test)
# selector = SelectKBest(score_func=mutual_info_classif, k=min(1500, X_train_scaled.shape[1]))
# X_train_selected = selector.fit_transform(X_train_scaled, y_train_res)
# X_test_selected = selector.transform(X_test_scaled)
# model = LogisticRegression(max_iter=500, solver='saga', class_weight='balanced')
# model.fit(X_train_selected, y_train_res)
# y_pred = (model.predict_proba(X_test_selected)[:, 1] >= 0.7).astype(int)
# print("\n📊 Binary Classification")
# print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
# print(f"Precision: {precision_score(y_test, y_pred):.2f}")
# print(f"Recall: {recall_score(y_test, y_pred):.2f}")
# print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")
# print(f"Review %: {np.mean(y_pred) * 100:.2f}%")
# print(f"Non-Review %: {(1 - np.mean(y_pred)) * 100:.2f}%")

# # === Multiclass Classification ===
# def map_review_class(x):
#     x = str(x).strip().lower()
#     if x == 'class-1': return 'great-review'
#     elif x in {'class-2', 'class-3'}: return 'decent-or-poor-review'
#     elif x == 'class-4': return 'spam-review'
#     return np.nan

# review_data = merged_data[merged_data['label'] == 1].copy()
# review_data['Review_Class'] = review_data['Rating'].apply(map_review_class)
# review_data = review_data.dropna(subset=['Review_Class'])

# X_text_cls = tfidf_vec.transform(review_data['text'])
# X_pos_cls = pos_vec.transform(review_data['POS'])
# sent_cls = np.array([[extract_sentiment(t) for t in review_data['text']]]).T
# subj_cls = np.array([[extract_subjectivity(t) for t in review_data['text']]]).T
# sent_len_cls, punc_cls, word_len_cls = readability_features(review_data)
# X_pattern_cls = review_data[[c for c in review_data.columns if c.startswith('pattern_')]].values
# X_url_cls = review_data[['has_review_url']].values
# X_cls = hstack([X_text_cls, X_pos_cls, sent_cls, subj_cls, sent_len_cls, punc_cls, word_len_cls, X_pattern_cls, X_url_cls])
# y_cls = review_data['Review_Class'].values

# X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, stratify=y_cls, test_size=0.3, random_state=42)
# X_train_cls, y_train_cls = SMOTE(random_state=42).fit_resample(X_train_cls, y_train_cls)
# scaler_cls = StandardScaler(with_mean=False)
# X_train_cls = scaler_cls.fit_transform(X_train_cls)
# X_test_cls = scaler_cls.transform(X_test_cls)
# selector_cls = SelectKBest(score_func=mutual_info_classif, k=min(1500, X_train_cls.shape[1]))
# X_train_cls = selector_cls.fit_transform(X_train_cls, y_train_cls)
# X_test_cls = selector_cls.transform(X_test_cls)
# model_cls = LogisticRegression(max_iter=1000, solver='saga', class_weight='balanced', multi_class='multinomial')
# model_cls.fit(X_train_cls, y_train_cls)
# y_pred_cls = model_cls.predict(X_test_cls)
# print("\n📊 Multiclass Classification")
# print(f"Accuracy: {accuracy_score(y_test_cls, y_pred_cls):.2f}")
# print(f"Precision: {precision_score(y_test_cls, y_pred_cls, average='macro'):.2f}")
# print(f"Recall: {recall_score(y_test_cls, y_pred_cls, average='macro'):.2f}")
# print(f"F1 Score: {f1_score(y_test_cls, y_pred_cls, average='macro'):.2f}")
# label_dist = Counter(y_pred_cls)
# total_preds = sum(label_dist.values())
# for label in ['great-review', 'decent-or-poor-review', 'spam-review']:
#     pct = (label_dist[label] / total_preds * 100) if label in label_dist else 0
#     print(f"{label.capitalize()} %: {pct:.2f}%")