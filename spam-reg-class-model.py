import pandas as pd
import numpy as np
import krippendorff
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.cluster import SpectralClustering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import spacy
import re
import os
from scipy.sparse import hstack
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.feature_selection import SelectKBest, mutual_info_regression

# === Rating Files ===
rating_files = [
    "rating/ratings-abhi-set1.csv",
    "rating/ratings-supi-set1.csv",
    "rating/ratings-vish-set1.csv",
    "rating/ratings-anu-set2.csv",
    "rating/ratings-pav-set2.csv",
    "rating/ratings-sum-set2.csv",
    "rating/ratings-naks-set3.csv",
    "rating/ratings-shrads-set3.csv",
    "rating/ratings-prah-set3.csv",
    "rating/ratings.csv",
    "rating/ratings1.csv",
    "rating/ratings2.csv"
]

# === Map categorical rating to numerical ===
def map_ratings_to_numerical(rating):
    rating_map = {
        "count-webshop": 1,
        "count-other": 2,
        "class-1": 3,
        "class-2": 4,
        "class-3": 5,
        "class-4": 6,
        "count-banner": 7,
        "count-broken": 8,
        "count-error": 9,
        "class-guide": 10,
        "others": 11
    }
    return rating_map.get(rating, np.nan)

# === Load & align 3 rating files ===
def load_and_align_ratings(files):
    all_ratings = []
    for i, file in enumerate(files):
        df = pd.read_csv(file, names=['Screenshot', 'Rating'])
        df['Rater'] = f'rater{i+1}'
        all_ratings.append(df)

    merged = all_ratings[0]
    for i, df in enumerate(all_ratings[1:], start=2):
        merged = pd.merge(merged, df, on='Screenshot', how='outer', suffixes=('', f'_{i}'))

    expected_columns = ['Screenshot']
    for i in range(1, 4):
        expected_columns.extend([f'Rating_rater{i}', f'Rater{i}'])

    merged.columns = expected_columns
    return merged

# === Compute Krippendorff’s Alpha across all sets ===
def compute_krippendorffs_alpha(rating_files_all):
    alpha_scores = []
    for i in range(0, len(rating_files_all), 3):
        files = rating_files_all[i:i+3]
        ratings = load_and_align_ratings(files)

        for r in range(1, 4):
            ratings[f'Rating_rater{r}'] = ratings[f'Rating_rater{r}'].apply(map_ratings_to_numerical)

        ratings_filtered = ratings.dropna(subset=[f'Rating_rater{r}' for r in range(1, 4)])
        ratings_matrix = ratings_filtered[[f'Rating_rater{r}' for r in range(1, 4)]].values.T
        alpha = krippendorff.alpha(reliability_data=ratings_matrix)

        print(f"Alpha for set {i//3 + 1}: {alpha:.4f}")
        alpha_scores.append(alpha)

    avg_alpha = np.mean(alpha_scores)
    print(f"\nAverage Krippendorff’s Alpha across all sets: {avg_alpha:.4f}")
    return alpha_scores, avg_alpha


compute_krippendorffs_alpha(rating_files)

# Load spaCy model for POS tagging
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 10000000  # Set a very high value to avoid memory issues with large texts

# Paths to your files
ground_truth_file = "rating/ground_truth_spam_with_url_final.csv"  # Ground truth with URLs and labels
#ground_truth_file = "rating/truth.csv"  # Ground truth with URLs and labels
warc_extracted_content_file = "rating/extracted_all_urls.csv"  # WARC extracted file with Text and POS_Tags
#warc_extracted_content_file = "rating/warc_extracted_content.csv"
data_file_to_predict = "warc_extracted_5_urls.csv"  # File with URLs to predict

from textblob import TextBlob
from scipy.sparse import hstack
import numpy as np
import re

# Clean text
def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = re.sub(r'(?i)(header|footer|copyright).*$', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Sentiment score
def extract_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Readability: sentence length and punctuation ratio
def readability_features(data):
    sentence_length = np.array([[len(text.split()) for text in data['text']]]).T
    punctuation_ratio = np.array([[
        sum(1 for char in text if char in '!?.,;:') / max(len(text.split()), 1)
        for text in data['text']
    ]]).T
    return sentence_length, punctuation_ratio

# Final feature extraction using TF-IDF + sentiment + readability
def extract_features(data, tfidf_vectorizer):
    # Cleaned TF-IDF features
    data['text_cleaned'] = data['text'].apply(preprocess_text)
    X_tfidf = tfidf_vectorizer.transform(data['text_cleaned'])

    # Readability
    #sentence_length, punctuation_ratio = readability_features(data)

    # Sentiment
    #sentiment = np.array([[extract_sentiment(text) for text in data['text_cleaned']]]).T

    # Combine features
    X_combined = hstack([X_tfidf])
    return X_combined

# Step 1: Load the ground truth file and warc extracted content file
ground_truth = pd.read_csv(ground_truth_file)
warc_data = pd.read_csv(warc_extracted_content_file)

# Step 2: Merge the ground truth with the WARC data based on URLs
merged_data = pd.merge(ground_truth, warc_data[['URL', 'text']], on='URL', how='left')

# Check if there are any URLs in ground truth that don't have corresponding data in the WARC file
missing_data = merged_data[merged_data['text'].isna()]
if not missing_data.empty:
    print("These URLs from ground truth are missing corresponding data in the WARC file:")
    #print(missing_data[['URL']])

# Preprocess and extract POS tags for the merged data
merged_data['text'] = merged_data['text'].apply(preprocess_text)  # Clean text

# Step 3: Prepare the Ground Truth Data for Training
# Use TfidfVectorizer to extract text features (text of the websites)
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english',ngram_range=(1, 2))
X_tfidf = tfidf_vectorizer.fit_transform(merged_data['text'].astype(str))

# Combine TF-IDF and POS features
X = hstack([X_tfidf])

# Labels (Review or Non-Review) based on the Rating column in ground truth
y = merged_data['Rating'].apply(lambda x: 1 if x in ["class-1", "class-2", "class-3", "class-4"] else 0).values  

# Step 4: Split Ground Truth Data into 70-30 for Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Training data size: {X_train.shape[0]}")
print(f"Testing data size: {X_test.shape[0]}")

smote_enn = SMOTEENN(random_state=42)
X_train_res, y_train_res = smote_enn.fit_resample(X_train, y_train)

# Scaling the data
scaler = StandardScaler(with_mean=False)  # we use `with_mean=False` because we are working with sparse matrices
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train the Logistic Regression model
model = LogisticRegression(max_iter=500, solver='saga', class_weight='balanced')
model.fit(X_train_scaled, y_train_res)

# Step 6: Evaluate the model on the test data
y_pred = model.predict(X_test_scaled)

# # Compute the Mean Squared Error (MSE)
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error (MSE) on Test Data: {mse:.4f}")

# Compute accuracy, precision, recall, F1 score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Step 7: Load the WARC extracted file for prediction (URLs to be predicted)
data_to_predict = pd.read_csv(data_file_to_predict)

# Step 8: Preprocess and extract POS tags for the WARC URLs
data_to_predict['text'] = data_to_predict['text'].apply(preprocess_text)  # Preprocess text from URLs

# Step 9: Prepare the data for prediction
X_predict_tfidf = tfidf_vectorizer.transform(data_to_predict['text'].astype(str))


# Combine TF-IDF and POS features for prediction
X_predict = hstack([X_predict_tfidf])

# Step 10: Predict Review/Non-Review for URLs in WARC file
y_predict = model.predict(X_predict)

# Add the predictions to the data dataframe
data_to_predict['Predicted_Review'] = y_predict

# Step 11: Calculate the percentage of Review and Non-Review URLs
total_urls = data_to_predict.shape[0]  # Total number of URLs

# Count the number of Review and Non-Review URLs
num_reviews = data_to_predict[data_to_predict['Predicted_Review'] == 1].shape[0]
num_non_reviews = data_to_predict[data_to_predict['Predicted_Review'] == 0].shape[0]

# Calculate percentages
review_percentage = (num_reviews / total_urls) * 100
non_review_percentage = (num_non_reviews / total_urls) * 100

# Print the results
print(f"Percentage of Review URLs: {review_percentage:.2f}%")
print(f"Percentage of Non-Review URLs: {non_review_percentage:.2f}%")


# train_review_data = merged_data[merged_data['Rating'].isin(['class-1', 'class-2', 'class-3', 'class-4'])]
# X_class_tfidf = tfidf_vectorizer.transform(train_review_data['text'])

# X_class = hstack([X_class_tfidf])
# y_class = train_review_data['Rating']

# # Train a classification model
# class_model = LogisticRegression(max_iter=1000, solver='saga', class_weight='balanced')
# class_model.fit(X_class, y_class)

# # Step 12: Predict the Class for Review URLs in WARC file
# review_data = data_to_predict[data_to_predict['Predicted_Review'] == 1]

# X_predict_class_tfidf = tfidf_vectorizer.transform(review_data['text'].astype(str))

# X_predict_class = hstack([X_predict_class_tfidf])

# # Predict the class for Review URLs
# y_predict_class = class_model.predict(X_predict_class)

# # Add the predicted classes to the dataframe
# data_to_predict.loc[data_to_predict['Predicted_Review'] == 1, 'Predicted_Class'] = y_predict_class

# # Step 13: For Non-Review URLs, set the class as 'others'
# data_to_predict['Predicted_Class'] = data_to_predict['Predicted_Class'].fillna('others')

# # Count the number of predictions for each class (1-4) and 'others'
# num_class_1 = data_to_predict[data_to_predict['Predicted_Class'] == 'class-1'].shape[0]
# num_class_2 = data_to_predict[data_to_predict['Predicted_Class'] == 'class-2'].shape[0]
# num_class_3 = data_to_predict[data_to_predict['Predicted_Class'] == 'class-3'].shape[0]
# num_class_4 = data_to_predict[data_to_predict['Predicted_Class'] == 'class-4'].shape[0]
# num_others = data_to_predict[data_to_predict['Predicted_Class'] == 'others'].shape[0]

# # Total number of predictions
# total_predictions = data_to_predict.shape[0]

# # Calculate percentages
# class_1_percentage = (num_class_1 / total_predictions) * 100
# class_2_percentage = (num_class_2 / total_predictions) * 100
# class_3_percentage = (num_class_3 / total_predictions) * 100
# class_4_percentage = (num_class_4 / total_predictions) * 100
# others_percentage = (num_others / total_predictions) * 100

# # Print the percentages for each class
# print(f"Percentage of Class-1 URLs: {class_1_percentage:.2f}%")
# print(f"Percentage of Class-2 URLs: {class_2_percentage:.2f}%")
# print(f"Percentage of Class-3 URLs: {class_3_percentage:.2f}%")
# print(f"Percentage of Class-4 URLs: {class_4_percentage:.2f}%")
# print(f"Percentage of 'others' URLs: {others_percentage:.2f}%")

# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# # Predict on ground truth to evaluate multiclass classification
# y_class_pred = class_model.predict(X_class)

# # Evaluate performance
# accuracy = accuracy_score(y_class, y_class_pred)
# precision = precision_score(y_class, y_class_pred, average='macro')  # Macro = equal weight per class
# recall = recall_score(y_class, y_class_pred, average='macro')
# f1 = f1_score(y_class, y_class_pred, average='macro')

# print("\n📊 Multiclass Classification Evaluation (on Ground Truth Reviews)")
# print(f" Accuracy:  {accuracy:.2f}")
# print(f" Precision: {precision:.2f}")
# print(f" Recall:    {recall:.2f}")
# print(f" F1 Score:  {f1:.2f}")

train_review_data = merged_data[merged_data['Rating'].isin(['class-1', 'class-2', 'class-3', 'class-4'])]
X_class_tfidf = tfidf_vectorizer.transform(train_review_data['text'])

X_class = hstack([X_class_tfidf])
y_class = train_review_data['Rating']

# ✅ Train/test split for multiclass model
X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42, stratify=y_class)

# ✅ Train a classification model on training set
class_model = LogisticRegression(max_iter=1000, solver='saga', class_weight='balanced')
class_model.fit(X_class_train, y_class_train)

# ✅ Evaluate on test set
y_class_pred_test = class_model.predict(X_class_test)
accuracy = accuracy_score(y_class_test, y_class_pred_test)
precision = precision_score(y_class_test, y_class_pred_test, average='macro', zero_division=0)
recall = recall_score(y_class_test, y_class_pred_test, average='macro', zero_division=0)
f1 = f1_score(y_class_test, y_class_pred_test, average='macro', zero_division=0)


print("\n📊 Multiclass Classification Evaluation (on Test Set)")
print(f" Accuracy:  {accuracy:.2f}")
print(f" Precision: {precision:.2f}")
print(f" Recall:    {recall:.2f}")
print(f" F1 Score:  {f1:.2f}")
total_predictions = data_to_predict.shape[0]
# ✅ Predict the Class for Review URLs in WARC file
review_data = data_to_predict[data_to_predict['Predicted_Review'] == 1]
X_predict_class_tfidf = tfidf_vectorizer.transform(review_data['text'].astype(str))
X_predict_class = hstack([X_predict_class_tfidf])
y_predict_class = class_model.predict(X_predict_class)
data_to_predict.loc[data_to_predict['Predicted_Review'] == 1, 'Predicted_Class'] = y_predict_class

# For Non-Review URLs, set the class as 'others'
data_to_predict['Predicted_Class'] = data_to_predict['Predicted_Class'].fillna('others')

# Class stats
for cls in ['class-1', 'class-2', 'class-3', 'class-4', 'others']:
    pct = (data_to_predict[data_to_predict['Predicted_Class'] == cls].shape[0] / total_predictions) * 100
    print(f"Percentage of {cls.capitalize()} URLs: {pct:.2f}%")

# Step 14: Save the result to a new file
# data_to_predict[['url', 'Predicted_Review', 'Predicted_Class']].to_csv('predicted_reviews_for_urls_classified.csv', index=False)

# print("Predictions completed and saved to 'predicted_reviews_for_urls_classified.csv'.")

# from sklearn.cluster import SpectralClustering
# from sklearn.metrics import accuracy_score, f1_score

# # Step 1: Filter only review data (class-1 to class-4)
# review_data = data_to_predict[data_to_predict['Predicted_Class'].isin(['class-1', 'class-2', 'class-3', 'class-4'])].copy()

# # Step 2: Assign Spam_Label_Classifier based on class-4
# review_data['Spam_Label_Classifier'] = review_data['Predicted_Class'].apply(lambda x: 1 if x == 'class-4' else 0)

# # review_data_gt = merged_data[merged_data['Rating'].isin(['class-1', 'class-2', 'class-3', 'class-4'])].copy()
# # review_data_gt['Spam_Label_GT'] = review_data_gt['Rating'].apply(lambda x: 1 if x == 'class-4' else 0)

# # Step 3: Extract features (TF-IDF + POS)
# X_cluster_tfidf = tfidf_vectorizer.transform(review_data['text'].astype(str))
# X_cluster_pos = pos_vectorizer.transform(review_data['POS_Tags'])
# X_features = hstack([X_cluster_tfidf, X_cluster_pos])

# # # Step 4: Apply Spectral Clustering directly on original data
# spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42)
# cluster_labels = spectral.fit_predict(X_features)
# review_data['Cluster_Label'] = cluster_labels

# # # Step 5: Identify which cluster is mostly spam (most class-4)
# cluster_spam_counts = review_data.groupby('Cluster_Label')['Spam_Label_Classifier'].sum()
# spam_cluster = cluster_spam_counts.idxmax()

# # # Step 6: Assign final Spam_Label based on cluster
# review_data['Spam_Label_Final'] = review_data['Cluster_Label'].apply(lambda x: 1 if x == spam_cluster else 0)

# # # Step 7: Compare clustering vs classifier
# y_true = review_data['Spam_Label_Classifier']
# y_pred = review_data['Spam_Label_Final']

# acc = accuracy_score(y_true, y_pred)
# f1 = f1_score(y_true, y_pred)

# print(f"\n Classifier vs Spectral Clustering Agreement")
# print(f" Accuracy: {acc:.2f}")
# print(f" F1 Score: {f1:.2f}")

# # # Step 8: Save and merge back into full dataset

# data_to_predict = data_to_predict.merge(
#     review_data[['url', 'Spam_Label_Final']],
#     on='url',
#     how='left'
# )

# # Step 9: Summary
# spam_count = review_data['Spam_Label_Final'].sum()
# non_spam_count = len(review_data) - spam_count

# print(f"\n✅ Spectral Clustering (pure) completed.")
# print(f" Spam (cluster with most class-4): {spam_count} ({100 * spam_count / len(review_data):.2f}%)")
# print(f" Non-spam (class-1/2/3): {non_spam_count} ({100 * non_spam_count / len(review_data):.2f}%)")
# review_data[['url','Predicted_Review', 'Predicted_Class']].to_csv("refined_spam_detection.csv", index=False)

# print(" Saved to 'refined_spam_detection.csv'")


# from sklearn.metrics import accuracy_score, f1_score

# # Agreement between classifier labels and clustering
# y_true = review_data['Spam_Label_Classifier']
# y_pred = review_data['Spam_Label_Final']

# accuracy = accuracy_score(y_true, y_pred)
# f1 = f1_score(y_true, y_pred)

# print(f"\n📊 Agreement Between Classifier and Clustering:")
# print(f" Accuracy: {accuracy:.2f}")
# print(f" F1 Score: {f1:.2f}")

# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split

# # Use final spam labels from clustering
# X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_features, review_data['Spam_Label_Final'], test_size=0.3, random_state=42)

# # Logistic Regression
# lr_model = LogisticRegression(max_iter=1000, solver='saga', class_weight='balanced')
# lr_model.fit(X_train_clf, y_train_clf)
# lr_preds = lr_model.predict(X_test_clf)

# lr_accuracy = accuracy_score(y_test_clf, lr_preds)
# lr_f1 = f1_score(y_test_clf, lr_preds)

# print(f"\n🔎 Logistic Regression Results (after clustering):")
# print(f" Accuracy: {lr_accuracy:.2f}")
# print(f" F1 Score: {lr_f1:.2f}")

# # SVM
# svm_model = SVC(kernel='linear', class_weight='balanced')
# svm_model.fit(X_train_clf, y_train_clf)
# svm_preds = svm_model.predict(X_test_clf)

# svm_accuracy = accuracy_score(y_test_clf, svm_preds)
# svm_f1 = f1_score(y_test_clf, svm_preds)

# print(f"\n🔎 SVM Results (after clustering):")
# print(f" Accuracy: {svm_accuracy:.2f}")
# print(f" F1 Score: {svm_f1:.2f}")

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
# Step 1: Normalize the ratings
merged_data['Rating'] = merged_data['Rating'].fillna('').astype(str).str.strip().str.lower()

# Step 2: Define review labels
review_labels = {'class-1', 'class-2', 'class-3', 'class-4'}

# Step 3: Filter rows that contain ANY of the review labels
def is_review_entry(rating):
    labels = [lbl.strip() for lbl in rating.split(',')]
    return any(lbl in review_labels for lbl in labels)

review_entries = merged_data[merged_data['Rating'].apply(is_review_entry)].copy()
print(f"✅ Number of review entries: {len(review_entries)}")

# Step 4: Map to numeric spam score
spam_class_map = {
    'class-1': 1,
    'class-2': 2,
    'class-3': 3,
    'class-4': 4
}

def extract_max_spam_score(rating_str):
    labels = [lbl.strip() for lbl in rating_str.split(',')]
    scores = [spam_class_map.get(lbl) for lbl in labels if lbl in spam_class_map]
    return max(scores) if scores else np.nan

review_entries['Spam_Score'] = review_entries['Rating'].apply(extract_max_spam_score)

# X_gt_tfidf = tfidf_vectorizer.transform(review_entries['text'].astype(str))
# sentiment = np.array([[extract_sentiment(text) for text in review_entries['text']]]).T
# sentence_length, punctuation_ratio = readability_features(review_entries)

# X_gt = hstack([X_gt_tfidf, sentence_length, punctuation_ratio, sentiment])
#X_gt = hstack([X_gt_tfidf])

# Preprocess text on-the-fly during prediction
# --- Step 2: Predict on review websites classified by the model ---
# TF-IDF features from ground truth review entries
X_gt_tfidf = tfidf_vectorizer.transform(review_entries['text'].astype(str))

# Readability + Sentiment features
sentiment = np.array([[extract_sentiment(text) for text in review_entries['text']]]).T
sentence_length, punctuation_ratio = readability_features(review_entries)

# Combine all features
X_gt = hstack([X_gt_tfidf])

# Target labels
y_gt = review_entries['Spam_Score'].values


# Split and train
#X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_gt, y_gt, test_size=0.4, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_gt, y_gt, test_size=0.1, random_state=42, stratify=y_gt
)

scaler = StandardScaler(with_mean=False)
X_train_scaled = scaler.fit_transform(X_train_reg)
X_test_scaled = scaler.transform(X_test_reg)

# Step 8: Feature selection (reduce to 1000 best features)
selector = SelectKBest(score_func=f_regression, k=1000)
X_train_selected = selector.fit_transform(X_train_scaled, y_train_reg)
X_test_selected = selector.transform(X_test_scaled)
# ridge = Ridge(alpha=1.0)
# ridge.fit(X_train_reg, y_train_reg)
# #ridge.fit(X_gt, y_gt)  # No split
# y_pred_reg = ridge.predict(X_test_reg)
for a in [0.01, 0.1, 1.0, 5.0, 10.0]:
    ridge = Ridge(alpha=a)
    ridge.fit(X_train_reg, y_train_reg)
    y_pred_reg = ridge.predict(X_test_reg)
    print(f"Alpha: {a}, MSE: {mean_squared_error(y_test_reg, y_pred_reg):.4f}, R²: {r2_score(y_test_reg, y_pred_reg):.2f}, Pearson: {np.corrcoef(y_pred_reg, y_test_reg)[0,1]:.2f}")

# Evaluation on ground truth test data
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"\n Ridge Regression Results on Ground Truth Review Data")
print(f" MSE: {mse:.4f}")
print(f" R² Score: {r2:.2f}")
print(f"Pearson r: {np.corrcoef(y_pred_reg, y_test_reg)[0,1]}")
print(f"Test Sample Size: {len(y_test_reg)}")

# --- Step 2: Predict on review websites classified by the model ---
# Predict on review websites classified by the model
review_predictions = data_to_predict[data_to_predict['Predicted_Review'] == 1].copy()

# Preprocess text
review_predictions['text'] = review_predictions['text'].apply(preprocess_text)

# TF-IDF
X_pred_tfidf = tfidf_vectorizer.transform(review_predictions['text'])

# Sentiment and Readability
sentiment_pred = np.array([[extract_sentiment(text) for text in review_predictions['text']]]).T
sentence_length_pred, punctuation_ratio_pred = readability_features(review_predictions)

# Combine all 1003 features (same as used in training)
X_pred = hstack([X_pred_tfidf])

# Now predict
spam_score_pred = ridge.predict(X_pred)


# Assign scores back to the main dataframe
data_to_predict.loc[data_to_predict['Predicted_Review'] == 1, 'Predicted_Spam_Score_Ridge'] = spam_score_pred

# Save final output
data_to_predict[['url', 'Predicted_Review', 'Predicted_Class', 'Predicted_Spam_Score_Ridge']].to_csv("refined_spam_detection.csv", index=False)
print("✅ Spam scores predicted and saved to 'refined_spam_detection.csv'.")

# --- Step 3: Plot actual vs predicted on test data ---
jitter = 0.05
x_jittered = y_test_reg + np.random.normal(0, jitter, size=len(y_test_reg))
y_jittered = y_pred_reg + np.random.normal(0, jitter, size=len(y_pred_reg))

slope, intercept = np.polyfit(y_test_reg, y_pred_reg, 1)
x_vals = np.array([1, 4])
y_vals = slope * x_vals + intercept

plt.figure(figsize=(8, 6))
sns.scatterplot(x=x_jittered, y=y_jittered, alpha=0.6)
plt.plot(x_vals, y_vals, color='blue', label='Regression Line')
plt.plot([1, 4], [1, 4], 'r--', label='Ideal')
plt.xlabel("Ground Truth Spam Score")
plt.ylabel("Predicted Spam Score")
plt.title(f"Ridge Regression (MSE: {mse:.2f}, R²: {r2:.2f})")
plt.legend()
plt.tight_layout()
plt.savefig("ridge_regression_actual_vs_predicted.png")
plt.close()

# --- Step 4: Histogram for predicted spam scores ---
plt.figure(figsize=(8, 6))
sns.histplot(spam_score_pred, kde=True, bins=30, color='blue', alpha=0.6)
plt.xlabel("Predicted Spam Score")
plt.title("Distribution of Predicted Spam Scores for Review URLs")
plt.tight_layout()
plt.savefig("predicted_spam_scores_histogram.png")
plt.close()

# --- Step 5: Confusion Matrix (Binary classifier) ---
# y_pred_class = model.predict(X_test_scaled)
# cm = confusion_matrix(y_test, y_pred_class)
# print("Confusion Matrix:")
# print(cm)
#https://kb.webis.de/services/slurm/user-guide.html
#https://huggingface.co/docs/transformers/en/model_doc/bert