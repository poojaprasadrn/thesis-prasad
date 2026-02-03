import os
import pickle
import pandas as pd
from fastwarc import ArchiveIterator
from resiliparse.parse.html import HTMLTree
from resiliparse.extract.html2text import extract_plain_text
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from elasticsearch import Elasticsearch, helpers
import requests

# Constants
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"
}
WARC_DIRECTORY = "/mnt/ceph/storage/data-in-progress/data-research/web-search/affiliate-serp-crawls"
EXTRACTED_CONTENT_CSV = "extracted_content.csv"
BEST_MODEL_FILE = "best_model.pkl"
BEST_VECTORIZER_FILE = "best_vectorizer.pkl"
CLASSIFIED_RESULTS_CSV = "classified_results.csv"
LIMIT = 10  # Limit the number of URLs to process

# Elasticsearch constants
ES_HOST = "https://elasticsearch.srv.webis.de/"
ES_USERNAME = "yili5634"
ES_PASSWORD = "KqAk50zmz80BSjtn"
ES_INDEX = "wstud_yili5634_classification_results"


def extract_content_from_warc(warc_directory, output_file, limit=LIMIT):
    """
    Extracts text content from WARC files and saves it to a CSV file.
    """
    data = []
    count = 0

    for root, _, files in os.walk(warc_directory):
        for file in files:
            if file.endswith('.warc') or file.endswith('.warc.gz'):
                warc_path = os.path.join(root, file)
                print(f"Processing WARC file: {warc_path}")

                with open(warc_path, 'rb') as stream:
                    for record in ArchiveIterator(stream):
                        if record.headers['WARC-Type'] == 'response':
                            url = record.headers.get('WARC-Target-URI', None)
                            if not url:
                                continue

                            try:
                                print(f"Fetching URL: {url}")
                                response = requests.get(url, headers=HEADERS, timeout=10)
                                response.raise_for_status()
                                html_content = response.text
                                html_tree = HTMLTree.parse(html_content)
                                plain_text = extract_plain_text(html_tree)
                                if plain_text:
                                    data.append({"url": url, "text": plain_text})
                                    count += 1
                                    if count >= limit:
                                        break
                            except Exception as e:
                                print(f"Error processing {url}: {e}")

            if count >= limit:
                break

    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} texts to {output_file}")


def evaluate_models(input_csv):
    """
    Train and evaluate multiple models and feature sets to find the best combination.
    """
    # Load dataset
    df = pd.read_csv(input_csv)

    # Create dummy labels for training
    df['label'] = [1 if i % 2 == 0 else 0 for i in range(len(df))]

    # Extract features and labels
    texts = df['text']
    labels = df['label']

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Define vectorizers and models
    vectorizers = {
        "TF": CountVectorizer(max_features=10000, ngram_range=(1, 2)),
        "TF-IDF": TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    }
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(),
        "SVM": SVC(probability=True, kernel='linear', random_state=42)
    }

    # Evaluate combinations
    best_score = 0
    best_model = None
    best_vectorizer = None

    for vec_name, vectorizer in vectorizers.items():
        X_train_vec = vectorizer.fit_transform(X_train)
        X_val_vec = vectorizer.transform(X_val)

        for model_name, model in models.items():
            print(f"Evaluating {model_name} with {vec_name}...")
            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_val_vec)
            score = accuracy_score(y_val, y_pred)
            print(f"Accuracy for {model_name} with {vec_name}: {score}")

            if score > best_score:
                best_score = score
                best_model = model
                best_vectorizer = vectorizer

    # Save the best model and vectorizer
    with open(BEST_MODEL_FILE, "wb") as mf:
        pickle.dump(best_model, mf)
    with open(BEST_VECTORIZER_FILE, "wb") as vf:
        pickle.dump(best_vectorizer, vf)

    print(f"Best Model: {best_model}")
    print(f"Best Vectorizer: {best_vectorizer}")
    print(f"Best Accuracy: {best_score}")


def classify_text(input_csv, vectorizer_file, model_file, output_file=CLASSIFIED_RESULTS_CSV):
    """
    Use the best trained model to classify text as AI or human-generated.
    """
    # Load the vectorizer and model
    with open(vectorizer_file, "rb") as vf:
        vectorizer = pickle.load(vf)
    with open(model_file, "rb") as mf:
        model = pickle.load(mf)

    # Load extracted text
    df = pd.read_csv(input_csv)

    # Vectorize text
    tfidf_features = vectorizer.transform(df['text'])

    # Predict
    probabilities = model.predict_proba(tfidf_features)[:, 1]  # Probability of being human-written
    predictions = (probabilities > 0.5).astype(int)

    # Add results to the DataFrame
    results_df = pd.DataFrame({
        'url': df['url'],
        'human_score': probabilities,
        'prediction': predictions
    })

    # Save results
    results_df.to_csv(output_file, index=False)
    print(f"Classification results saved to {output_file}")

    return results_df


def upload_to_elasticsearch(es_host, es_username, es_password, classification_df, index_name):
    """
    Upload classification results to Elasticsearch.
    """
    es = Elasticsearch(
        hosts=[es_host],
        basic_auth=(es_username, es_password),
        verify_certs=True
    )

    # Convert DataFrame to Elasticsearch actions
    actions = [
        {
            "_index": index_name,
            "_source": result
        }
        for result in classification_df.to_dict(orient='records')
    ]

    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name)
        print(f"Index '{index_name}' created in Elasticsearch.")

    helpers.bulk(es, actions)
    print(f"Classification results uploaded to Elasticsearch index '{index_name}'.")


if __name__ == "__main__":
    # Step 1: Extract content
    print("Extracting content from WARC files...")
    extract_content_from_warc(WARC_DIRECTORY, EXTRACTED_CONTENT_CSV, limit=LIMIT)

    # Step 2: Train and save the best model and vectorizer
    print("Evaluating models to find the best one...")
    evaluate_models(EXTRACTED_CONTENT_CSV)

    # Step 3: Classify text using the best model
    print("Classifying text using the best model...")
    classified_results = classify_text(EXTRACTED_CONTENT_CSV, BEST_VECTORIZER_FILE, BEST_MODEL_FILE)

    # Step 4: Upload results to Elasticsearch
    print("Uploading results to Elasticsearch...")
    upload_to_elasticsearch(ES_HOST, ES_USERNAME, ES_PASSWORD, classified_results, ES_INDEX)

from pyspark import SparkContext
from pyspark.sql import SparkSession
from fastwarc import ArchiveIterator
from bs4 import BeautifulSoup
from transformers import pipeline
import os
from resiliparse.parse.html import HTMLTree
from resiliparse.extract.html2text import extract_plain_text
from fastwarc import ArchiveIterator

#Set up Spark session and context
spark = SparkSession.builder \
    .appName("WARC AI Content Detection") \
    .getOrCreate()
sc = spark.sparkContext

#Load the transformer model pipeline for text classification
classifier = pipeline("text-classification", model="roberta-large-openai-detector")

# Function to extract URLs and plain text content from a WARC file
def fetch_content_from_warc(warc_file_path):
    url_content_pairs = []
    try:
        with open(warc_file_path, 'rb') as stream:
            for record in ArchiveIterator(stream):
                if record.headers['WARC-Type'] == 'response':
                    content_type = record.http_headers.get('Content-Type', '')
                    if 'text/html' in content_type:
                        url = record.headers['WARC-Target-URI']
                        html_content = record.reader.read().decode('utf-8', errors='ignore')

                        # Parse HTML content with Resiliparse
                        try:
                            html_tree = HTMLTree.parse(html_content)
                            plain_text = extract_plain_text(html_tree)
                        except Exception as e:
                            print(f"Failed to parse HTML for {url}: {e}")
                            plain_text = None

                        if plain_text and url.startswith(("http://", "https://")):
                            url_content_pairs.append((url, plain_text))
    except Exception as e:
        print(f"Error reading WARC file {warc_file_path}: {e}")

    return url_content_pairs


#Function to classify the content of the URL
def classify_content(url_content_pair):
    try:
        url, content = url_content_pair
        if content:
            # Split content into chunks of 500 characters each to analyze the whole page
            chunk_size = 500
            chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
            ai_count = 0
            human_count = 0
            
            for chunk in chunks:
                result = classifier(chunk)
                label = result[0]['label']
                
                if label == 'LABEL_1':
                    ai_count += 1
                elif label == 'LABEL_0':
                    human_count += 1

            # Determine final label based on counts
            if ai_count > 0 and human_count > 0:
                detection_result = 'Mixed'
            elif ai_count > 0:
                detection_result = 'AI Detected'
            elif human_count > 0:
                detection_result = 'Human Detected'
            else:
                detection_result = 'Unknown'
        else:
            detection_result = 'Content Fetch Failed'
        return url, detection_result
    except Exception as e:
        print(f"Error classifying URL {url}: {e}")
        return url, 'Classification Failed'

#Read WARC files into an RDD
warc_directory = '/mnt/ceph/storage/data-in-progress/data-research/web-search/affiliate-serp-crawls'

def find_warc_files(directory):
    warc_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.warc') or file.endswith('.warc.gz'):
                warc_files.append(os.path.join(root, file))
    return warc_files

warc_files = find_warc_files(warc_directory)

warc_rdd = sc.parallelize(warc_files)

#Extract URLs and content from WARC files using Spark map operation
url_content_rdd = warc_rdd.flatMap(lambda warc_file: fetch_content_from_warc(warc_file)).take(50)

#Classify content locally
results = [result for result in (classify_content(url_content) for url_content in url_content_rdd) if result is not None and result[1] != 'Content Fetch Failed']

#Convert to DataFrame for saving
results_df = spark.createDataFrame(results, ["URL", "Detection Result"])

#Save results to CSV
results_df.coalesce(1).write.csv("/mnt/ceph/storage/data-tmp/current/yili5634/classification_results.csv", header=True, mode='overwrite')


spark.stop()
