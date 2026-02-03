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
spam_categories = {"class-4", "count-banner", "count-broken", "count-error"}
non_spam_categories = {"class-1", "class-2", "class-3", "count-webshop", "count-guide", "count-other"}

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
ground_truth_file = "rating/ground_truth_spam_with_url.csv"
processed_pos_file = "processed_pos_tags.csv"
data = pd.read_csv(data_file)
unique_urls = data['url'].unique()[:20]  # Select the first 100 unique URLs
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


X_combined = np.hstack([ tfidf_matrix.toarray(), pos_count_matrix.toarray()])
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
# ground_truth['URL'] = ground_truth['URL'].str.strip().str.lower().str.rstrip('/')

# Ground Truth sampling for 40 entries (train-test split 70-30)
ground_truth = ground_truth.sample(n=61, random_state=42)


# Split ground_truth into 70% training and 30% testing (28 train, 12 test)
gt_train, gt_test = train_test_split(ground_truth, test_size=0.3, random_state=42)

# # Get train and test URLs from ground_truth (though we won't use them for matching)
# train_urls = gt_train['URL'].tolist()
# test_urls = gt_test['URL'].tolist()

# Get corresponding indices for the training and testing URLs in ground_truth (not used for data)
train_indices = gt_train.index.tolist()
test_indices = gt_test.index.tolist()

# Now, directly use ground_truth for training labels
y_train = gt_train['Label'].apply(lambda x: 1 if x == "Spam" else 0).values
y_test = gt_test['Label'].apply(lambda x: 1 if x == "Spam" else 0).values

# The data itself is being used for predictions, not the ground truth URLs
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



# import numpy as np
# import pandas as pd
# import re
# from sklearn.model_selection import train_test_split
# import spacy
# from tqdm import tqdm
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.preprocessing import normalize
# from sklearn.cluster import KMeans, SpectralClustering
# from sklearn.manifold import TSNE
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import LinearSVC
# from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
# import matplotlib.pyplot as plt
# import os
# import math

# # ====== Helper Functions ======
# def split_and_process_text(text):
#     """
#     Splits large text into manageable chunks and processes each for POS tagging.
#     Args:
#         text (str): The input text.
#         chunk_size (int): Maximum length of each chunk.
#     Returns:
#         str: Combined POS tags from all chunks.
#     """
#     if pd.isna(text) or not isinstance(text, str):
#         return ""

#     text = re.sub(r'\s+', ' ', text.strip().lower())  # Clean text
#     text = re.sub(r'[^\w\s]', '', text) 
    
#     doc = nlp(text)

#     # Extract POS tags
#     pos_tags = [token.pos_ for token in doc if token.is_alpha]

#     return " ".join(pos_tags)

# def _ngrams(tokens, n):
#     """Generate n-grams from token sequences."""
#     return [' '.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

# def _analyzer(tokens):
#     """
#     Combine unigrams, bigrams, trigrams, and POS frequency counts.
#     """
#     unigrams = tokens
#     bigrams = _ngrams(tokens, 2)
#     trigrams = _ngrams(tokens, 3)
#     return unigrams + bigrams + trigrams

# def _ceil10(n):
#     return int(math.ceil(n / 10) * 10)

# def _floor10(n):
#     return int(math.floor(n / 10) * 10)

# def _config_axis(ax, data):
#     ax.set_ticks(np.linspace(_floor10(np.min(data)), _ceil10(np.max(data)), 3))
#     ax.set_ticklabels([])
#     ax.pane.fill = False
#     ax.pane.set_linewidth(0.5)
#     ax.pane.set_alpha(1.0)
#     ax.pane.set_edgecolor(plt.rcParams['grid.color'])

# def plot_scatter(X_3d, y, title, legend_title, outfile=None, subsample=1.0):
#     """
#     Plots a 3D scatter plot of clusters with transparent or outlined markers.

#     Args:
#         X_3d (numpy.ndarray): 3D embeddings for data points.
#         y (numpy.ndarray): Cluster labels (Spam/Non-Spam).
#         title (str): Plot title.
#         legend_title (str): Legend title.
#         outfile (str): Filepath to save the plot (optional).
#         subsample (float): Proportion of data to subsample for visualization.
#     """
#     fig = plt.figure(figsize=(6, 6))
#     ax = fig.add_subplot(projection='3d', computed_zorder=True)
#     ax.view_init(30, 125)

#     # Subsample the data if needed
#     if subsample < 1.0:
#         indices = np.random.choice(X_3d.shape[0], int(X_3d.shape[0] * subsample), replace=False)
#         X_3d_sampled = X_3d[indices]
#         y_sampled = y[indices]
#     else:
#         X_3d_sampled = X_3d
#         y_sampled = y

#     # Plot clusters with transparent markers
#     labels = ['Non-spam', 'Spam']
#     colors = ['blue', 'orange']
#     for i, label in enumerate(labels):
#         scatter_data = X_3d_sampled[y_sampled == i]
#         ax.scatter(
#             scatter_data[:, 0], scatter_data[:, 1], scatter_data[:, 2],
#             label=f'{label}',
#             color=colors[i],
#             alpha=0.4,  # Increase transparency
#             s=25
#         )

#     # Project points to the XY plane
#     scatter_data_projected = np.copy(X_3d_sampled)
#     scatter_data_projected[:, 2] = np.min(X_3d[:, 2])
#     ax.scatter(
#         scatter_data_projected[:, 0], scatter_data_projected[:, 1], scatter_data_projected[:, 2],
#         c='lightgray', s=15, lw=0, alpha=0.4
#     )

#     # Configure axes
#     _config_axis(ax.axes.xaxis, X_3d[:, 0])
#     _config_axis(ax.axes.yaxis, X_3d[:, 1])
#     _config_axis(ax.axes.zaxis, X_3d[:, 2])
#     ax.axes.margins(0, 0, 0, tight=False)

#     # Customize legend to match your example
#     legend = ax.legend(loc="upper right", fontsize=10, title=legend_title)
#     legend.get_title().set_fontsize(10)  # Set font size for legend title
#     plt.title(title, fontsize=12)

#     fig.canvas.draw()
#     plt.tight_layout()

#     # Save or show the plot
#     if outfile:
#         directory = os.path.dirname(outfile)
#         if directory:  # Only attempt to create a directory if it exists
#             os.makedirs(directory, exist_ok=True)
#         plt.savefig(outfile, bbox_inches='tight')
#     plt.show()

# # ====== Load spaCy Model ======
# nlp = spacy.load("en_core_web_sm")
# nlp.max_length = 10000000  # Increase max_length for spaCy

# # ====== Step 1: Process and Save POS Tags ======
# data_file = "warc_extracted_5_urls.csv"  # Replace with your file path
# processed_file = "processed_data_with_pos.csv"

# # Check if processed data already exists
# if os.path.exists(processed_file):
#     print("Processed file found. Loading...")
#     data = pd.read_csv(processed_file)
# else:
#     print("Processing text for POS tagging...")
#     data = pd.read_csv(data_file)
#     data['POS_Tags'] = [split_and_process_text(text) for text in tqdm(data['text'], desc="Processing Texts")]
#     data.to_csv(processed_file, index=False)
#     print(f"Processed data saved to {processed_file}.")

# invalid_rows = data[data['POS_Tags'].isna()]
# print(f"Number of invalid rows after POS tagging: {len(invalid_rows)}")


# # ====== Step 2: Feature Extraction ======
# # Convert POS_Tags into lists of tokens
# data['POS_Tokens'] = data['POS_Tags'].str.split()

# # Extract POS n-grams and build feature matrix
# print("Extracting POS n-grams...")
# pos_vectorizer = CountVectorizer(analyzer=lambda x: _analyzer(x), max_features=300)
# pos_count_matrix = pos_vectorizer.fit_transform(data['POS_Tokens'])
# X = normalize(pos_count_matrix.toarray(), norm='l1', axis=1)
# print(f"Feature matrix shape: {X.shape}")

# # Print POS n-grams by frequency
# pos_freqs = np.array(pos_count_matrix.sum(axis=0)).flatten()
# print('POS n-grams by frequency:')
# print([p[0].replace(' ', '+') for p in sorted(pos_vectorizer.vocabulary_.items(), key=lambda x: pos_freqs[x[1]], reverse=True)])


# # ====== Load Ground Truth ======
# ground_truth_file = "ground_truth_results.csv"
# print("Loading ground truth...")
# ground_truth = pd.read_csv(ground_truth_file)

# # Normalize URLs for consistency
# data['url'] = data['url'].str.strip().str.lower().str.rstrip('/')
# ground_truth['url'] = ground_truth['url'].str.strip().str.lower().str.rstrip('/')
# ground_truth.set_index('url', inplace=True)

# print(f"Ground truth loaded with {len(ground_truth)} entries.")


# # ====== Split Data ======
# print("Splitting data into labeled and non-labeled sets...")
# urls = data['url'].tolist()

# labeled_indices = [i for i, url in enumerate(urls) if url in ground_truth.index]
# labeled_X = X[labeled_indices]
# labeled_y = ground_truth.loc[[urls[i] for i in labeled_indices], 'Spam_Label'].apply(lambda x: 1 if x == "Spam" else 0).values
# labeled_urls = [urls[i] for i in labeled_indices]

# non_labeled_indices = [i for i, url in enumerate(urls) if url not in ground_truth.index]
# non_labeled_X = X[non_labeled_indices]
# non_labeled_urls = [urls[i] for i in non_labeled_indices]

# # ====== Split Ground Truth (70-30 split) ======
# print("Splitting labeled data (70-30 split)...")
# gt_X_train, gt_X_test, gt_y_train, gt_y_test, gt_urls_train, gt_urls_test = train_test_split(
#     labeled_X, labeled_y, labeled_urls, test_size=0.3, random_state=42
# )

# # Combine Labeled Training Data and Non-Labeled Data
# X_train = np.vstack([gt_X_train, non_labeled_X])
# y_train = np.hstack([gt_y_train, np.full(len(non_labeled_X), -1)])  # -1 for non-labeled
# urls_train = gt_urls_train + non_labeled_urls

# # Strictly use only labeled ground truth test data
# X_test = gt_X_test
# y_test = gt_y_test
# urls_test = gt_urls_test

# # Verify Final Data Shapes
# print(f"Training data shape: {X_train.shape}")
# print(f"Testing data shape: {X_test.shape}")
# print(f"Ground truth test size: {len(gt_X_test)}")


# # ====== Perform Spectral Clustering ======
# print("Performing Spectral Clustering...")
# spectral_model = SpectralClustering(
#     n_clusters=2,
#     affinity='nearest_neighbors',
#     random_state=42,
#     n_jobs=-1
# )
# y_train = spectral_model.fit_predict(X_train)

# if np.sum(y_train) / y_train.shape[0] < 0.5:
#     y_train = 1 - y_train

# print(f'Non-Spam: {np.sum(y_train)} ({np.sum(y_train) / len(y_train):.2%})')
# print(f'Spam: {y_train.shape[0] - np.sum(y_train)}')



# # ====== Visualize with t-SNE ======
# print("Reducing data to 3D with t-SNE...")
# X_3d = TSNE(n_components=3, random_state=29, init='random', method='barnes_hut', n_jobs=-1).fit_transform(np.vstack((X_train, X_test)))
# X_train_3d = X_3d[:X_train.shape[0], :]
# X_test_3d = X_3d[X_train.shape[0]:, :]

# # Plot Training Set
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(projection='3d')
# colors = ['blue' if label == 1 else 'orange' for label in y_train]
# ax.scatter(X_train_3d[:, 0], X_train_3d[:, 1], X_train_3d[:, 2], c=colors, label='Training Data', alpha=0.4)
# plt.title("Training Data Clusters")
# plt.show()

# # Plot Test Set
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(projection='3d')
# colors_test = ['blue' if label == 1 else 'orange' for label in y_test]
# ax.scatter(X_test_3d[:, 0], X_test_3d[:, 1], X_test_3d[:, 2], c=colors_test, label='Test Data', alpha=0.4)
# plt.title("Test Data Clusters (Ground Truth)")
# plt.show()

# # ====== Save Results ======
# output_file = "spam_detection_results.csv"
# results = pd.DataFrame({
#     'url': urls_train + urls_test,
#     'label': ['Non-Spam' if label == 1 else 'Spam' for label in y_train.tolist() + y_test.tolist()]
# })
# results.to_csv(output_file, index=False)
# print(f"Results saved to {output_file}.")

# # ====== Evaluate Models ======
# print("Evaluating models...")
# for clf in [LogisticRegression(), LinearSVC(dual='auto')]:
#     print(f'{clf.__class__.__name__}:-----')
    
#     clf.fit(X_train[y_train > -1], y_train[y_train > -1])
#     y_pred = clf.predict(X_test)
#     y_scores = clf.decision_function(X_test) if hasattr(clf, 'decision_function') else None
    
#     print(f'Accuracy (balanced): {balanced_accuracy_score(y_test, y_pred):.2f}')
#     if y_scores is not None:
#         print(f'AUROC: {roc_auc_score(y_test, y_scores):.2f}')
    
#     print('\nNon-Spam:')
#     print(f'Precision: {precision_score(y_test, y_pred):.2f}')
#     print(f'Recall: {recall_score(y_test, y_pred):.2f}')
#     print(f'F1: {f1_score(y_test, y_pred):.2f}')
    
#     print('\nSpam:')
#     print(f'Precision: {precision_score(y_test, y_pred, pos_label=0):.2f}')
#     print(f'Recall: {recall_score(y_test, y_pred, pos_label=0):.2f}')
#     print(f'F1: {f1_score(y_test, y_pred, pos_label=0):.2f}\n\n')
-------------------------------
#spam reg code with pos tags 
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

# Function to preprocess and clean the text before extracting POS tags
def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text.strip().lower())  # Remove extra spaces and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Function to extract POS tags from the text using spaCy
def extract_pos_tags(text):
    doc = nlp(text)
    pos_tags = [token.pos_ for token in doc if token.is_alpha]  # Extract POS tags for alphabetic tokens
    return " ".join(pos_tags)  # Return POS tags as a space-separated string

# Function to extract sentiment score using TextBlob
def extract_sentiment(text):
    sentiment = TextBlob(text).sentiment.polarity  # Returns the sentiment polarity
    return sentiment

# Function to calculate sentence length and punctuation ratio
def readability_features(data):
    sentence_length = np.array([[len(text.split()) for text in data['text']]]).T  # Word count (sentence length)
    punctuation_ratio = np.array([[sum(1 for char in text if char in '!?.,;:') / len(text.split()) for text in data['text']]]).T  # Punctuation ratio
    return sentence_length, punctuation_ratio

# Function to extract features and concatenate them
def extract_features(data, tfidf_vectorizer, pos_vectorizer):
    # TF-IDF Features
    X_tfidf = tfidf_vectorizer.transform(data['text'].astype(str))
    
    # POS Features
    X_pos = pos_vectorizer.transform(data['POS_Tags'])
    
    # Readability Features
    sentence_length, punctuation_ratio = readability_features(data)
    
    # Sentiment Features
    sentiment = np.array([[extract_sentiment(text) for text in data['text']]]).T
    
    # Combine all the features into a single matrix
    X = hstack([X_tfidf, X_pos, sentence_length, punctuation_ratio, sentiment])
    return X

# Step 1: Load the ground truth file and warc extracted content file
ground_truth = pd.read_csv(ground_truth_file)
warc_data = pd.read_csv(warc_extracted_content_file)

# Step 2: Merge the ground truth with the WARC data based on URLs
merged_data = pd.merge(ground_truth, warc_data[['URL', 'text', 'POS_Tags']], on='URL', how='left')

# Check if there are any URLs in ground truth that don't have corresponding data in the WARC file
missing_data = merged_data[merged_data['text'].isna()]
if not missing_data.empty:
    print("These URLs from ground truth are missing corresponding data in the WARC file:")
    #print(missing_data[['URL']])

# Preprocess and extract POS tags for the merged data
merged_data['text'] = merged_data['text'].apply(preprocess_text)  # Clean text
merged_data['POS_Tags'] = merged_data['text'].apply(extract_pos_tags)

# Step 3: Prepare the Ground Truth Data for Training
# Use TfidfVectorizer to extract text features (text of the websites)
tfidf_vectorizer = TfidfVectorizer(max_features=300, stop_words='english',ngram_range=(1, 3))
X_tfidf = tfidf_vectorizer.fit_transform(merged_data['text'].astype(str))

# POS features using CountVectorizer
pos_vectorizer = CountVectorizer(analyzer=lambda x: x.split(), max_features=150)
X_pos = pos_vectorizer.fit_transform(merged_data['POS_Tags'])

# Combine TF-IDF and POS features
X = hstack([X_tfidf, X_pos])

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
model = LogisticRegression(max_iter=1000, solver='saga', class_weight='balanced')
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
data_to_predict['text'] = data_to_predict['url'].apply(preprocess_text)  # Preprocess text from URLs
data_to_predict['POS_Tags'] = data_to_predict['text'].apply(extract_pos_tags)  # Extract POS tags

# Step 9: Prepare the data for prediction
X_predict_tfidf = tfidf_vectorizer.transform(data_to_predict['url'].astype(str))
X_predict_pos = pos_vectorizer.transform(data_to_predict['POS_Tags'])

# Combine TF-IDF and POS features for prediction
X_predict = hstack([X_predict_tfidf, X_predict_pos])

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


train_review_data = merged_data[merged_data['Rating'].isin(['class-1', 'class-2', 'class-3', 'class-4'])]
X_class_tfidf = tfidf_vectorizer.transform(train_review_data['text'])
X_class_pos = pos_vectorizer.transform(train_review_data['POS_Tags'])
X_class = hstack([X_class_tfidf, X_class_pos])
y_class = train_review_data['Rating']

# Train a classification model
class_model = LogisticRegression(max_iter=1000, solver='saga', class_weight='balanced')
class_model.fit(X_class, y_class)

# Step 12: Predict the Class for Review URLs in WARC file
review_data = data_to_predict[data_to_predict['Predicted_Review'] == 1]

X_predict_class_tfidf = tfidf_vectorizer.transform(review_data['url'].astype(str))
X_predict_class_pos = pos_vectorizer.transform(review_data['POS_Tags'])

X_predict_class = hstack([X_predict_class_tfidf, X_predict_class_pos])

# Predict the class for Review URLs
y_predict_class = class_model.predict(X_predict_class)

# Add the predicted classes to the dataframe
data_to_predict.loc[data_to_predict['Predicted_Review'] == 1, 'Predicted_Class'] = y_predict_class

# Step 13: For Non-Review URLs, set the class as 'others'
data_to_predict['Predicted_Class'] = data_to_predict['Predicted_Class'].fillna('others')

# Count the number of predictions for each class (1-4) and 'others'
num_class_1 = data_to_predict[data_to_predict['Predicted_Class'] == 'class-1'].shape[0]
num_class_2 = data_to_predict[data_to_predict['Predicted_Class'] == 'class-2'].shape[0]
num_class_3 = data_to_predict[data_to_predict['Predicted_Class'] == 'class-3'].shape[0]
num_class_4 = data_to_predict[data_to_predict['Predicted_Class'] == 'class-4'].shape[0]
num_others = data_to_predict[data_to_predict['Predicted_Class'] == 'others'].shape[0]

# Total number of predictions
total_predictions = data_to_predict.shape[0]

# Calculate percentages
class_1_percentage = (num_class_1 / total_predictions) * 100
class_2_percentage = (num_class_2 / total_predictions) * 100
class_3_percentage = (num_class_3 / total_predictions) * 100
class_4_percentage = (num_class_4 / total_predictions) * 100
others_percentage = (num_others / total_predictions) * 100

# Print the percentages for each class
print(f"Percentage of Class-1 URLs: {class_1_percentage:.2f}%")
print(f"Percentage of Class-2 URLs: {class_2_percentage:.2f}%")
print(f"Percentage of Class-3 URLs: {class_3_percentage:.2f}%")
print(f"Percentage of Class-4 URLs: {class_4_percentage:.2f}%")
print(f"Percentage of 'others' URLs: {others_percentage:.2f}%")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Predict on ground truth to evaluate multiclass classification
y_class_pred = class_model.predict(X_class)

# Evaluate performance
accuracy = accuracy_score(y_class, y_class_pred)
precision = precision_score(y_class, y_class_pred, average='macro')  # Macro = equal weight per class
recall = recall_score(y_class, y_class_pred, average='macro')
f1 = f1_score(y_class, y_class_pred, average='macro')

print("\n📊 Multiclass Classification Evaluation (on Ground Truth Reviews)")
print(f" Accuracy:  {accuracy:.2f}")
print(f" Precision: {precision:.2f}")
print(f" Recall:    {recall:.2f}")
print(f" F1 Score:  {f1:.2f}")


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

X_gt_tfidf = tfidf_vectorizer.transform(review_entries['text'].astype(str))
X_gt_pos = pos_vectorizer.transform(review_entries['POS_Tags'])
X_gt = hstack([X_gt_tfidf, X_gt_pos])
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
review_predictions = data_to_predict[data_to_predict['Predicted_Review'] == 1].copy()

# Use correct features: these must be based on cleaned "text", not "url"
X_pred_tfidf = tfidf_vectorizer.transform(review_predictions['text'].astype(str))
X_pred_pos = pos_vectorizer.transform(review_predictions['POS_Tags'])
X_pred = hstack([X_pred_tfidf, X_pred_pos])

# Predict spam scores
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
