import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import spacy
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from sklearn.cluster import SpectralClustering
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import krippendorff
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import os
import math
from glob import glob
from collections import Counter

# Load spaCy Model
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 10000000
spam_categories = {"class-4", "count-banner", "count-broken", "count-error"}
non_spam_categories = {"class-1", "class-2", "class-3", "count-webshop", "count-guide", "count-other"}

def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text.strip().lower())
    text = re.sub(r'[^\w\s]', '', text)
    doc = nlp(text)
    pos_tags = [token.pos_ for token in doc if token.is_alpha]
    return " ".join(pos_tags)

def generate_ngrams(tokens, n):
    return [' '.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

def analyzer(tokens):
    unigrams = tokens
    bigrams = generate_ngrams(tokens, 2)
    trigrams = generate_ngrams(tokens, 3)
    return unigrams + bigrams + trigrams

import requests
from bs4 import BeautifulSoup
import numpy as np

# SEO Feature Extraction Function
def get_seo_features(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # SEO features as numpy arrays
        meta_description = soup.find('meta', attrs={'name': 'description'})
        meta_description_length = len(meta_description['content']) if meta_description else 0

        title = soup.find('title')
        title_length = len(title.get_text()) if title else 0

        h1_tags = soup.find_all('h1')
        h1_count = len(h1_tags)

        images = soup.find_all('img')
        total_images = len(images)
        images_with_alt = sum(1 for img in images if img.get('alt'))

        internal_links = [link['href'] for link in soup.find_all('a', href=True) if link['href'].startswith('/')]
        internal_links_count = len(internal_links)

        outbound_links = [link['href'] for link in soup.find_all('a', href=True) if not link['href'].startswith('/')]
        outbound_links_count = len(outbound_links)

        # URL length
        url_length = len(url)

        # Return features as numpy array
        return np.array([meta_description_length, title_length, h1_count, total_images, images_with_alt, 
                         internal_links_count, outbound_links_count, url_length])
    except Exception as e:
        # Handle any errors with default values (e.g., 0 for all features)
        return np.array([0, 0, 0, 0, 0, 0, 0, 0])

# Define Spam and Non-Spam categories
spam_categories = {"class-4"}
# "count-banner", "count-broken", "count-error"
non_spam_categories = {"class-1", "class-2", "class-3"}
# "count-webshop", "count-guide", "count-other"
# Function to map the rating to Spam (1) or Non-Spam (0)
def map_to_spam_nonspam(rating):
    if rating in spam_categories:
        return 1  # Spam
    elif rating in non_spam_categories:
        return 0  # Non-Spam
    else:
        return np.nan  # Return NaN if rating is not in either category

# Specify the path to your 'ratings' folder
ratings_folder = 'rating'  # Update this if your 'ratings' folder is in a different location

# List of your rating files
rating_files = ['ratings.csv', 'ratings1.csv', 'ratings2.csv']
file_paths = [os.path.join(ratings_folder, file) for file in rating_files]

# Check if the files exist in the 'ratings' folder
print(f"Found files: {file_paths}")

# Read all rating files
y_test_goldstandard = []
for d in tqdm(file_paths, desc='Loading truth labels'):
    print(f"Loading file: {d}")
    df = pd.read_csv(d, names=['Screenshot', 'Rating'])  # Assuming CSV has 'Screenshot' and 'Rating' columns
    # Map each rating to Spam (1) or Non-Spam (0) based on the category
    df['Rating'] = df['Rating'].apply(map_to_spam_nonspam)
    y_test_goldstandard.append(df)

# If no files are found, exit
if not y_test_goldstandard:
    print("No files found in the ratings folder.")
    exit()

# Concatenate all rating data
y_test_goldstandard = pd.concat(y_test_goldstandard)

# Drop any rows where the mapped rating is NaN
y_test_goldstandard = y_test_goldstandard.dropna(subset=['Rating'])

# Count how many times each screenshot was rated
screenshot_counts = y_test_goldstandard.groupby('Screenshot').size()

# Identify screenshots rated by all three raters
screenshots_all_three = screenshot_counts[screenshot_counts == 3].index.tolist()

# Identify screenshots rated by different raters (only 2 ratings)
screenshots_different_raters = screenshot_counts[screenshot_counts == 2].index.tolist()

# Group ratings by screenshot
grouped_labels = y_test_goldstandard.groupby('Screenshot')['Rating'].apply(list)

# Separate the ratings for analysis
all_three_raters = grouped_labels.loc[screenshots_all_three]  # Screenshots rated by all 3 raters
different_raters = grouped_labels.loc[screenshots_different_raters]  
# Compute Krippendorff's Alpha for screenshots rated by all three raters
if not all_three_raters.empty:
    alpha_all_three = krippendorff.alpha(reliability_data=all_three_raters.apply(
        lambda x: pd.Series(x)).transpose())
    print(f'Agreement (Krippendorff\'s α) for screenshots rated by all three raters: {alpha_all_three:.2f}')
else:
    print("No screenshots were rated by all three raters.")

# Inspect disagreements for screenshots rated by all three raters
disagreements = all_three_raters[all_three_raters.apply(lambda x: len(set(x)) > 1)]  # Screenshots with disagreements

# Print the disagreements and their ratings for each rater
if not disagreements.empty:
    print("Disagreements in ratings (Screenshots with different labels):")
    for screenshot, ratings in disagreements.items():
        print(f"Screenshot: {screenshot}, Ratings: {ratings}")

# Apply majority voting for screenshots rated by two raters
if not different_raters.empty:
    # Apply majority vote (if one rater is missing, take the average and round)
    majority_voting = different_raters.apply(lambda x: np.mean(x).round())
    print(f'Majority Voting Results for screenshots rated by two raters:')
    print(f'Non-Spam: {sum(majority_voting == 0)}')
    print(f'Spam: {sum(majority_voting == 1)}')

# Final report
print(f"Total Screenshots Rated by All Three Raters: {len(all_three_raters)}")
# print(f"Total Screenshots Rated by Different Raters (2 raters): {len(different_raters)}")
print(f"Total Unique Screenshots Rated: {len(grouped_labels)}")



# Load Data
data_file = "warc_extracted_5_urls.csv"
ground_truth_file = "rating/truth.csv"
#ground_truth_file = "rating/ground_truth_spam_with_url.csv"
processed_pos_file = "processed_pos_tags.csv"
data = pd.read_csv(data_file)
unique_urls = data['url'].unique() # Select the first 100 unique URLs
data = data[data['url'].isin(unique_urls)] 
ground_truth = pd.read_csv(ground_truth_file)

# Check if POS tags have already been processed
if os.path.exists(processed_pos_file):
    print("Loading preprocessed POS tags...")
    processed_data = pd.read_csv(processed_pos_file)
    data['POS_Tags'] = processed_data['POS_Tags']
else:
    print("Processing POS tags for the first time...")
    data['POS_Tags'] = [preprocess_text(text) for text in tqdm(data['text'], desc="Processing Texts")]
    processed_data = data[['POS_Tags']]
    processed_data.to_csv(processed_pos_file, index=False)
    print(f"Saved processed POS tags to {processed_pos_file}")

tfidf_vectorizer = TfidfVectorizer(max_features=300, ngram_range=(1, 3), stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['text'].astype(str))
lexical_features = np.array([[len(text), len(text.split()), sum(c.isdigit() for c in text), sum(c in "!?." for c in text)] for text in data['text'].astype(str)])
scaler = MinMaxScaler()
lexical_features = scaler.fit_transform(lexical_features)
data['POS_Tags'] = data['POS_Tags'].fillna("").apply(lambda x: x if isinstance(x, str) else "")
data['POS_Tokens'] = data['POS_Tags'].str.split()

print("\n🧐 Checking Non-Empty POS_Tokens...")
valid_tokens_count = data['POS_Tokens'].dropna().apply(len).sum()
print(f"Total valid POS tokens: {valid_tokens_count}")
if valid_tokens_count == 0:
    print("⚠️ Error: No valid tokens found for POS Vectorization. Skipping CountVectorizer.")
    pos_count_matrix = None
else:
    print("✅ Proceeding with POS Vectorization...")
    pos_vectorizer = CountVectorizer(analyzer=lambda x: analyzer(x), max_features=400)
    pos_count_matrix = pos_vectorizer.fit_transform(data['POS_Tokens'])

readability_features = np.array([[text.count(' ')/len(text) if len(text) > 0 else 0, sum(c in '!?.,' for c in text)/len(text) if len(text) > 0 else 0] for text in data['text'].astype(str)])
readability_features = MinMaxScaler().fit_transform(readability_features)
syntactic_complexity = np.array([[text.count(' and ') + text.count(' but ') + text.count(' because ') for text in data['text'].astype(str)]]).T
noun_verb_ratio = np.array([[text.count('NOUN') / (text.count('VERB') + 1) for text in data['POS_Tags']]]).T
stopword_ratio = np.array([[sum(1 for token in text.split() if token.lower() in nlp.Defaults.stop_words) / (len(text.split()) + 1) for text in data['text'].astype(str)]]).T
content_density = np.array([[len(set(text.split())) / (len(text.split()) + 1) for text in data['text'].astype(str)]]).T
avg_word_length = np.array([[sum(len(word) for word in text.split()) / (len(text.split()) + 1) for text in data['text'].astype(str)]]).T
word_diversity = np.array([[len(set(text.split())) / max(1, len(text.split())) for text in data['text'].astype(str)]]).T
punctuation_ratio = np.array([[sum(c in '!?.,' for c in text) / max(1, len(text)) for text in data['text'].astype(str)]]).T
sentence_length_variance = np.array([[np.var([len(sentence) for sentence in text.split('.')]) if '.' in text else 0 for text in data['text'].astype(str)]]).T
type_token_ratio = np.array([[len(set(text.split())) / (len(text.split()) + 1) for text in data['text'].astype(str)]]).T
pos_diversity = np.array([[len(set(text.split())) / max(1, len(text.split())) for text in data['POS_Tags']]]).T
sentence_count = np.array([[text.count('.') + text.count('!') + text.count('?') for text in data['text'].astype(str)]]).T
uppercase_ratio = np.array([[sum(1 for c in text if c.isupper()) / max(1, len(text)) for text in data['text'].astype(str)]]).T
#seo_features = np.array([get_seo_features(url) for url in data['url'].astype(str)]).T  
# Ensure seo_features has the same number of rows as data
# seo_feature_dict = {url: get_seo_features(url) for url in data['url'].unique()}  # Get SEO features per unique URL

# # Create an array that maps SEO features to all rows
# seo_features_mapped = np.array([seo_feature_dict[url] for url in data['url']])  # Expand to match rows in data


X_combined = np.hstack([ tfidf_matrix.toarray(), pos_count_matrix.toarray(),noun_verb_ratio, content_density, avg_word_length, pos_diversity, sentence_count, uppercase_ratio])
#noun_verb_ratio, content_density, avg_word_length, pos_diversity, sentence_count, uppercase_ratio,seo_features_mapped,
X_combined = StandardScaler().fit_transform(X_combined)
X_combined = PCA(n_components=150).fit_transform(X_combined)
X = X_combined

# Print Feature Matrix Shape
print(f"Feature matrix shape: {X.shape}")

# Print POS n-grams by frequency
pos_freqs = np.array(pos_count_matrix.sum(axis=0)).flatten()
print('Most common POS n-grams:')
common_pos = sorted(pos_vectorizer.vocabulary_.items(), key=lambda x: pos_freqs[x[1]], reverse=True)[:20]
print([p[0].replace(' ', '+') for p in common_pos])

# Normalize URLs
data['url'] = data['url'].str.strip().str.lower().str.rstrip('/')
ground_truth['URL'] = ground_truth['URL'].str.strip().str.lower().str.rstrip('/')
# Load the ground truth file (ensure that this file is correctly loaded)
# Define the class mappings
non_spam_classes = ["class-1", "class-2", "class-3"]
spam_classes = ["class-4"]

# Filter Non-Spam and Spam websites
non_spam_reviews = ground_truth[ground_truth['Rating'].isin(non_spam_classes)]
spam_reviews = ground_truth[ground_truth['Rating'].isin(spam_classes)]

# Count how many reviews fall into Non-Spam and Spam
print(f"\nTotal Non-Spam Reviews: {len(non_spam_reviews)}")
print(f"Total Spam Reviews: {len(spam_reviews)}")


# Ground Truth sampling for 40 entries (train-test split 70-30)
#ground_truth = ground_truth.sample(n=61, random_state=42)


# Split ground_truth into 70% training and 30% testing (28 train, 12 test)
#gt_train, gt_test = train_test_split(ground_truth, test_size=0.3, random_state=42)

# Get train and test URLs from ground_truth (though we won't use them for matching)
#train_urls = gt_train['URL'].tolist()
#test_urls = gt_test['URL'].tolist()

# Get corresponding indices for the training and testing URLs in ground_truth (not used for data)
#train_indices = gt_train.index.tolist()
#test_indices = gt_test.index.tolist()

# Now, directly use ground_truth for training labels
#y_train = gt_train['Label'].apply(lambda x: 1 if x == "Spam" else 0).values
#y_test = gt_test['Label'].apply(lambda x: 1 if x == "Spam" else 0).values

# The data itself is being used for predictions, not the ground truth URLs
#X_train = X[train_indices]  # Feature matrix for training
#X_test = X[test_indices]    # Feature matrix for testing

# Print the shape of the training and testing data
#print(f"Training data shape: {X_train.shape}")
#print(f"Testing data shape: {X_test.shape}")

filtered_reviews = ground_truth[ground_truth['Rating'].isin(non_spam_classes + spam_classes)]

# Now split the filtered data into training and testing sets (70-30)
gt_train, gt_test = train_test_split(filtered_reviews, test_size=0.4, random_state=42)

# Get the indices for train and test sets
train_indices = gt_train.index.tolist()
test_indices = gt_test.index.tolist()

# Convert the training labels to 1 (Spam) and 0 (Non-Spam)
y_train = gt_train['Label'].apply(lambda x: 1 if x == "Spam" else 0).values
y_test = gt_test['Label'].apply(lambda x: 1 if x == "Spam" else 0).values

# Assuming the feature matrix X is already created
X_train = X[train_indices]  # Feature matrix for training
X_test = X[test_indices]    # Feature matrix for testing

# Print the shape of the training and testing data
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Perform Spectral Clustering (just to demonstrate a clustering example)
print("Performing spectral clustering...")
spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', n_neighbors=3, random_state=42)
predicted_clusters = spectral.fit_predict(X)  # Using the full dataset to make predictions
spam_count = np.sum(predicted_clusters)
non_spam_count = len(predicted_clusters) - spam_count
print(f"Non-Spam: {non_spam_count} ({non_spam_count/len(predicted_clusters)*100:.2f}%)")
print(f"Spam: {spam_count} ({spam_count/len(predicted_clusters)*100:.2f}%)")

smote = SMOTE(sampling_strategy='auto', k_neighbors=2,random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train models
print("Training models...")
models = [LogisticRegression(class_weight='balanced'), LinearSVC(class_weight='balanced', dual='auto')]

for model in models:
    model.fit(X_resampled, y_resampled)
    print(f'{model.__class__.__name__}:-----')
    y_pred = model.predict(X_test)
    y_scores = model.decision_function(X_test) if hasattr(model, 'decision_function') else None
    print(f'Accuracy (balanced): {balanced_accuracy_score(y_test, y_pred):.2f}')
    if y_scores is not None:
        print(f'AUROC: {roc_auc_score(y_test, y_scores):.2f}')
    print("\nNon-Spam:")
    print(f'Precision: {precision_score(y_test, y_pred, pos_label=0):.2f}')
    print(f'Recall: {recall_score(y_test, y_pred, pos_label=0):.2f}')
    print(f'F1: {f1_score(y_test, y_pred, pos_label=0):.2f}')
    print("\nSpam:")
    print(f'Precision: {precision_score(y_test, y_pred, pos_label=1):.2f}')
    print(f'Recall: {recall_score(y_test, y_pred, pos_label=1):.2f}')
    print(f'F1: {f1_score(y_test, y_pred, pos_label=1):.2f}\n\n')


# Generate 3D t-SNE Visualization for the training data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_3d = TSNE(n_components=2, perplexity=10, learning_rate=200, max_iter=300, random_state=42).fit_transform(X_scaled)

# Save Graphs
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

print("Generating Ground Truth Data Visualization...")
plt.figure(figsize=(8, 6))
plt.scatter(X_3d[train_indices, 0], X_3d[train_indices, 1], c=y_train, cmap='coolwarm', alpha=1.0, edgecolors='black', linewidths=0.5)
plt.title("Ground Truth Data Visualization (2D)")
plt.legend(['Spam', 'Non-Spam'], loc='upper right', fontsize=10, title='Cluster Labels')
plt.savefig(os.path.join(plot_dir, "ground_truth_2d.jpg"), format="jpg")
plt.close()
print("Ground Truth Data Visualization (2D) saved as ground_truth_2d.jpg")

print("Generating Ground Truth + Predicted Data Visualization...")
plt.figure(figsize=(8, 6))
plt.scatter(X_3d[:, 0], X_3d[:, 1], c=predicted_clusters, cmap='coolwarm', alpha=0.6, edgecolors='black', linewidths=0.5)
plt.title("Ground Truth + Predicted Data Visualization (2D)")
plt.legend(['Non-Spam', 'Spam'], loc='upper right', fontsize=10, title='Cluster Labels')
plt.savefig(os.path.join(plot_dir, "ground_truth_predicted_2d.jpg"), format="jpg")
plt.close()
print("Ground Truth + Predicted Data Visualization (2D) saved as ground_truth_predicted_2d.jpg")





# import pandas as pd
# import numpy as np
# import krippendorff
# from imblearn.over_sampling import SMOTE
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
# from sklearn.cluster import SpectralClustering
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# import spacy
# import re
# import os
# from scipy.sparse import hstack
# from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
# from textblob import TextBlob


# # Define the paths to your rating files
# rating_files = ["rating/ratings1.csv", "rating/ratings2.csv", "rating/ratings.csv"]

# # Function to map ratings to Review (1) and Non-Review (0)
# def map_to_review_nonreview(rating):
#     review_classes = ["class-1", "class-2", "class-3", "class-4"]  # Define review categories
#     return 1 if rating in review_classes else 0

# # Load and align ratings by Screenshot
# def load_and_align_ratings(rating_files):
#     all_ratings = []
    
#     # Load all the rating files and align them by Screenshot
#     for i, file in enumerate(rating_files):
#         df = pd.read_csv(file, names=['Screenshot', 'Rating'])  # Assuming CSV has 'Screenshot' and 'Rating' columns
#         df['Rater'] = f'rater{i+1}'  # Add a column to track the rater
#         all_ratings.append(df)

#     # Merge all the ratings based on the Screenshot column
#     merged_ratings = all_ratings[0]
#     for i, df in enumerate(all_ratings[1:], start=2):
#         merged_ratings = pd.merge(merged_ratings, df, on='Screenshot', how='outer', suffixes=('', f'_{i}'))

#     # Now, we have a merged DataFrame with ratings from all raters aligned by Screenshot
#     # We need to clean the data by ensuring columns are properly named
#     merged_ratings.columns = ['Screenshot', 'Rating_rater1', 'Rater1', 'Rating_rater2', 'Rater2', 'Rating_rater3', 'Rater3']
    
#     return merged_ratings

# # Function to map categorical ratings to numerical values
# def map_ratings_to_numerical(rating):
#     rating_map = {
#         "count-webshop": 1,
#         "count-other": 2,
#         "class-1": 3,
#         "class-2": 4,
#         "class-3": 5,
#         "class-4": 6,
#         "count-banner": 7,
#         "count-broken": 8,
#         "count-error": 9,
#         "class-guide": 10,  # Added new category for class-guide
#         "others": 11         # Example of a possible "others" category
#     }
#     return rating_map.get(rating, np.nan)  # Default to NaN for ratings that don't exist in the map

# # Load and align ratings
# ratings = load_and_align_ratings(rating_files)

# # Map the ratings to numerical values
# ratings['Rating_rater1'] = ratings['Rating_rater1'].apply(map_ratings_to_numerical)
# ratings['Rating_rater2'] = ratings['Rating_rater2'].apply(map_ratings_to_numerical)
# ratings['Rating_rater3'] = ratings['Rating_rater3'].apply(map_ratings_to_numerical)

# # Handle any missing values (if any ratings are missing)
# ratings = ratings.fillna(np.nan)  # Or use some other strategy (like filling with a specific value)

# # Filter out rows with missing ratings (where at least one rater has not rated)
# ratings_filtered = ratings.dropna(subset=['Rating_rater1', 'Rating_rater2', 'Rating_rater3'])

# # Display the filtered ratings where all raters have rated
# #print("\nRatings given by all raters:")
# #print(ratings_filtered[['Screenshot', 'Rating_rater1', 'Rating_rater2', 'Rating_rater3']])

# # Compute Krippendorff's Alpha
# def compute_krippendorffs_alpha(ratings):
#     # We only need the rating columns (rater1, rater2, rater3)
#     ratings_matrix = ratings[['Rating_rater1', 'Rating_rater2', 'Rating_rater3']].values.T  # Transpose to make each column a rater's ratings
#     alpha = krippendorff.alpha(reliability_data=ratings_matrix)
#     print(f"Krippendorff's Alpha: {alpha:.2f}")

# # Compute Krippendorff's alpha
# compute_krippendorffs_alpha(ratings)


# # Load spaCy model for POS tagging
# nlp = spacy.load("en_core_web_sm")
# nlp.max_length = 10000000  # Set a very high value to avoid memory issues with large texts

# # Paths to your files
# ground_truth_file = "rating/ground_truth_spam_with_url.csv"  # Ground truth with URLs and labels
# #ground_truth_file = "rating/truth.csv"  # Ground truth with URLs and labels
# warc_extracted_content_file = "rating/warc_extracted_content.csv"  # WARC extracted file with Text and POS_Tags
# data_file_to_predict = "warc_extracted_5_urls.csv"  # File with URLs to predict

# # Function to preprocess and clean the text before extracting POS tags
# def preprocess_text(text):
#     if pd.isna(text) or not isinstance(text, str):
#         return ""
#     text = re.sub(r'\s+', ' ', text.strip().lower())  # Remove extra spaces and convert to lowercase
#     text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
#     return text

# # Function to extract POS tags from the text using spaCy
# def extract_pos_tags(text):
#     doc = nlp(text)
#     pos_tags = [token.pos_ for token in doc if token.is_alpha]  # Extract POS tags for alphabetic tokens
#     return " ".join(pos_tags)  # Return POS tags as a space-separated string

# # Function to extract sentiment score using TextBlob
# def extract_sentiment(text):
#     sentiment = TextBlob(text).sentiment.polarity  # Returns the sentiment polarity
#     return sentiment

# # Function to calculate sentence length and punctuation ratio
# def readability_features(data):
#     sentence_length = np.array([[len(text.split()) for text in data['text']]]).T  # Word count (sentence length)
#     punctuation_ratio = np.array([[sum(1 for char in text if char in '!?.,;:') / len(text.split()) for text in data['text']]]).T  # Punctuation ratio
#     return sentence_length, punctuation_ratio

# # Function to extract features and concatenate them
# def extract_features(data, tfidf_vectorizer, pos_vectorizer):
#     # TF-IDF Features
#     X_tfidf = tfidf_vectorizer.transform(data['text'].astype(str))
    
#     # POS Features
#     X_pos = pos_vectorizer.transform(data['POS_Tags'])
    
#     # Readability Features
#     sentence_length, punctuation_ratio = readability_features(data)
    
#     # Sentiment Features
#     sentiment = np.array([[extract_sentiment(text) for text in data['text']]]).T
    
#     # Combine all the features into a single matrix
#     X = hstack([X_tfidf, X_pos, sentence_length, punctuation_ratio, sentiment])
#     return X

# # Step 1: Load the ground truth file and warc extracted content file
# ground_truth = pd.read_csv(ground_truth_file)
# warc_data = pd.read_csv(warc_extracted_content_file)

# # Step 2: Merge the ground truth with the WARC data based on URLs
# merged_data = pd.merge(ground_truth, warc_data[['URL', 'text', 'POS_Tags']], on='URL', how='left')

# # Check if there are any URLs in ground truth that don't have corresponding data in the WARC file
# missing_data = merged_data[merged_data['text'].isna()]
# if not missing_data.empty:
#     print("These URLs from ground truth are missing corresponding data in the WARC file:")
#     print(missing_data[['URL']])

# # Preprocess and extract POS tags for the merged data
# merged_data['text'] = merged_data['text'].apply(preprocess_text)  # Clean text
# merged_data['POS_Tags'] = merged_data['text'].apply(extract_pos_tags)

# # Step 3: Prepare the Ground Truth Data for Training
# # Use TfidfVectorizer to extract text features (text of the websites)
# tfidf_vectorizer = TfidfVectorizer(max_features=300, stop_words='english')
# X_tfidf = tfidf_vectorizer.fit_transform(merged_data['text'].astype(str))

# # POS features using CountVectorizer
# pos_vectorizer = CountVectorizer(analyzer=lambda x: x.split(), max_features=150)
# X_pos = pos_vectorizer.fit_transform(merged_data['POS_Tags'])

# # Combine TF-IDF and POS features
# X = hstack([X_tfidf, X_pos])

# # Labels (Review or Non-Review) based on the Rating column in ground truth
# y = merged_data['Rating'].apply(lambda x: 1 if x in ["class-1", "class-2", "class-3", "class-4"] else 0).values  

# # Step 4: Split Ground Truth Data into 70-30 for Training and Testing
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# print(f"Training data size: {X_train.shape[0]}")
# print(f"Testing data size: {X_test.shape[0]}")

# smote = SMOTE(random_state=42)
# X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# # Scaling the data
# scaler = StandardScaler(with_mean=False)  # we use `with_mean=False` because we are working with sparse matrices
# X_train_scaled = scaler.fit_transform(X_train_res)
# X_test_scaled = scaler.transform(X_test)
# # Step 5: Train the Logistic Regression model
# model = LogisticRegression(max_iter=1000, solver='saga', class_weight='balanced')
# model.fit(X_train_scaled, y_train_res)

# # Step 6: Evaluate the model on the test data
# y_pred = model.predict(X_test_scaled)

# # Compute the Mean Squared Error (MSE)
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error (MSE) on Test Data: {mse:.4f}")

# # Compute accuracy, precision, recall, F1 score
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)

# print(f"Accuracy: {accuracy:.2f}")
# print(f"Precision: {precision:.2f}")
# print(f"Recall: {recall:.2f}")
# print(f"F1 Score: {f1:.2f}")

# # Step 7: Load the WARC extracted file for prediction (URLs to be predicted)
# data_to_predict = pd.read_csv(data_file_to_predict)

# # Step 8: Preprocess and extract POS tags for the WARC URLs
# data_to_predict['text'] = data_to_predict['url'].apply(preprocess_text)  # Preprocess text from URLs
# data_to_predict['POS_Tags'] = data_to_predict['text'].apply(extract_pos_tags)  # Extract POS tags

# # Step 9: Prepare the data for prediction
# X_predict_tfidf = tfidf_vectorizer.transform(data_to_predict['url'].astype(str))
# X_predict_pos = pos_vectorizer.transform(data_to_predict['POS_Tags'])

# # Combine TF-IDF and POS features for prediction
# X_predict = hstack([X_predict_tfidf, X_predict_pos])

# # Step 10: Predict Review/Non-Review for URLs in WARC file
# y_predict = model.predict(X_predict)

# # Add the predictions to the data dataframe
# data_to_predict['Predicted_Review'] = y_predict

# # Step 11: Calculate the percentage of Review and Non-Review URLs
# total_urls = data_to_predict.shape[0]  # Total number of URLs

# # Count the number of Review and Non-Review URLs
# num_reviews = data_to_predict[data_to_predict['Predicted_Review'] == 1].shape[0]
# num_non_reviews = data_to_predict[data_to_predict['Predicted_Review'] == 0].shape[0]

# # Calculate percentages
# review_percentage = (num_reviews / total_urls) * 100
# non_review_percentage = (num_non_reviews / total_urls) * 100

# # Print the results
# print(f"Percentage of Review URLs: {review_percentage:.2f}%")
# print(f"Percentage of Non-Review URLs: {non_review_percentage:.2f}%")

# train_review_data = merged_data[merged_data['Rating'].isin(['class-1', 'class-2', 'class-3', 'class-4'])]
# X_class_tfidf = tfidf_vectorizer.transform(train_review_data['text'])
# X_class_pos = pos_vectorizer.transform(train_review_data['POS_Tags'])
# X_class = hstack([X_class_tfidf, X_class_pos])
# y_class = train_review_data['Rating']

# # Train a classification model
# class_model = LogisticRegression(max_iter=1000, solver='saga', class_weight='balanced')
# class_model.fit(X_class, y_class)

# # Step 12: Predict the Class for Review URLs in WARC file
# review_data = data_to_predict[data_to_predict['Predicted_Review'] == 1]

# X_predict_class_tfidf = tfidf_vectorizer.transform(review_data['url'].astype(str))
# X_predict_class_pos = pos_vectorizer.transform(review_data['POS_Tags'])

# X_predict_class = hstack([X_predict_class_tfidf, X_predict_class_pos])

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

# # Step 14: Save the result to a new file
# #data_to_predict[['url', 'Predicted_Review', 'Predicted_Class']].to_csv('predicted_reviews_for_urls_classified.csv', index=False)

# #print("Predictions completed and saved to 'predicted_reviews_for_urls_classified.csv'.")
# from sklearn.cluster import SpectralClustering
# from scipy.sparse import hstack

# # Filter only review data
# review_data = data_to_predict[data_to_predict['Predicted_Review'] == 1].copy()

# # Extract TF-IDF and POS features
# X_cluster_tfidf = tfidf_vectorizer.transform(review_data['text'].astype(str))
# X_cluster_pos = pos_vectorizer.transform(review_data['POS_Tags'])
# X_cluster = hstack([X_cluster_tfidf, X_cluster_pos])

# # Perform Spectral Clustering
# spectral = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42)
# clusters = spectral.fit_predict(X_cluster)

# # Temporarily assign cluster labels
# review_data['Temp_Cluster'] = clusters

# # Determine which cluster has more class-4 → that’s spam
# class4_counts = review_data.groupby('Temp_Cluster')['Predicted_Class'].apply(lambda x: (x == 'class-4').sum())
# spam_cluster = class4_counts.idxmax()

# # Assign final Spam_Label
# review_data['Spam_Label'] = review_data['Temp_Cluster'].apply(lambda x: 1 if x == spam_cluster else 0)
# review_data.drop(columns=['Temp_Cluster'], inplace=True)

# # Merge back into main dataframe
# data_to_predict = data_to_predict.merge(
#     review_data[['url', 'Spam_Label']],
#     on='url',
#     how='left'
# )

# # Compute percentages (only reviews with clustering labels)
# total = review_data.shape[0]
# spam_count = (review_data['Spam_Label'] == 1).sum()
# non_spam_count = (review_data['Spam_Label'] == 0).sum()
# spam_percent = (spam_count / total) * 100
# non_spam_percent = (non_spam_count / total) * 100

# # Save clustered review data
# review_data[['url', 'text', 'Predicted_Class', 'Spam_Label']].to_csv("spectral_spam_reviews_only.csv", index=False)

# # Output summary
# print(f"\ Spectral Clustering (Reviews Only) completed.")
# print(f" Spam (class-4 cluster): {spam_percent:.2f}%")
# print(f" Non-Spam (class-1/2/3 cluster): {non_spam_percent:.2f}%")
# print(" Saved review clustering output to 'spectral_spam_reviews_only.csv'")

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.manifold import TSNE
# from sklearn.preprocessing import StandardScaler

# # Load clustered review-only data
# df = pd.read_csv("spectral_spam_reviews_only.csv")

# # Prepare the feature set for t-SNE
# # Since the original features were not stored, we simulate with TF-IDF + POS features
# # For visualization, let's use TF-IDF of the text (as a proxy)
# from sklearn.feature_extraction.text import TfidfVectorizer

# tfidf_vectorizer = TfidfVectorizer(max_features=100)
# X_tfidf = tfidf_vectorizer.fit_transform(df['text'].astype(str))

# # Reduce dimensions with t-SNE for visualization
# tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=500)
# X_tsne = tsne.fit_transform(X_tfidf.toarray())

# # Add the t-SNE results to the DataFrame
# df['tsne_1'] = X_tsne[:, 0]
# df['tsne_2'] = X_tsne[:, 1]

# # Plot
# plt.figure(figsize=(10, 7))
# sns.scatterplot(
#     x='tsne_1',
#     y='tsne_2',
#     hue='Spam_Label',
#     palette={0: 'blue', 1: 'red'},
#     data=df,
#     alpha=0.6
# )
# plt.title("t-SNE Visualization of Spam (Red) vs Non-Spam (Blue) - Predicted Reviews")
# plt.xlabel("t-SNE Component 1")
# plt.ylabel("t-SNE Component 2")
# plt.legend(title="Spam Label", labels=["Non-Spam", "Spam"])
# plt.grid(True)
# plt.tight_layout()
# plt.show()
