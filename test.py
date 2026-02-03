import os
import pickle
import re
from elasticsearch import Elasticsearch, helpers
import pandas as pd
from pyspark import SparkConf, SparkContext
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from fastwarc import ArchiveIterator
from resiliparse.parse.html import HTMLTree
from resiliparse.extract.html2text import extract_plain_text
from datetime import datetime
from publicsuffixlist import PublicSuffixList
from adblockparser import AdblockRules  # Ad blocker integration
import requests

# Spark configuration

conf = (
    SparkConf()
    #.setMaster('k8s://https://k8s.srv.webis.de')  # Kubernetes master URL
    .setMaster('local[*]')
    #.setAppName('AIDetection')     # Application name
    .set('spark.executor.instances', 2)           # 2 executors to reduce memory pressure
    .set('spark.executor.memory', '10g')          # Memory per executor adjusted to fit within limits
    .set('spark.executor.cores', '1')             # Use 1 core per executor to reduce CPU load
    .set('spark.kubernetes.executor.request.cores', '1') # Request 1 core per executor
    .set('spark.kubernetes.executor.limit.cores', '1')       # Set cores per executor to 1 to match the CPU limit
    .set('spark.driver.memory', '8g')             # Reduce driver memory to 8GB
    .set('spark.kubernetes.container.image', 'registry.webis.de/code-teaching/theses/thesis-prasad:v1.0')  # Docker image
    .set('spark.kubernetes.namespace', 'spark-jobs')  # Kubernetes namespace
    .set('spark.kubernetes.authenticate.driver.serviceAccountName', 'spark')  # Service account
    .set('spark.kubernetes.container.image.pullSecrets', 'webis-registry-credentials')  # Image pull secret
    .set('spark.kubernetes.executor.podCreationTimeout', '600s')  # Increase executor pod creation timeout
    .set("spark.kubernetes.driver.annotation.sidecar.istio.io/inject", "false")
    .set("spark.kubernetes.executor.annotation.sidecar.istio.io/inject", "false")
    .set('spark.kubernetes.driver.volumes.hostPath.storage.options.path', '/mnt/ceph/storage')
    .set('spark.kubernetes.driver.volumes.hostPath.storage.mount.path', '/mnt/ceph/storage')
    .set('spark.kubernetes.executor.volumes.hostPath.storage.options.path', '/mnt/ceph/storage')
    .set('spark.kubernetes.executor.volumes.hostPath.storage.mount.path', '/mnt/ceph/storage')
)

# Initialize Spark context
sc = SparkContext(conf=conf)

# Constants
HEADERS = {"User-Agent": "Mozilla/5.0"}
WARC_DIRECTORY = "/mnt/ceph/storage/data-in-progress/data-research/web-search/affiliate-serp-crawls"
MODEL_FILE = "best_model.pkl"
VECTORIZER_FILE = "best_vectorizer.pkl"
CLASSIFIED_RESULTS_CSV = "classified_results.csv"
TRAINING_DATA = "training_data.csv"
ES_HOST = "https://elasticsearch.srv.webis.de/"
ES_USERNAME = "yili5634"
ES_PASSWORD = "KqAk50zmz80BSjtn"
ES_INDEX = "wstud_yili5634_classification_results"
EXTRACTED_CONTENT_CSV = "warc_extracted_content.csv"

# Initialize PublicSuffixList
psl = PublicSuffixList()

# Path to the shared storage for filter files
EASYLIST_PATH = "easylist.txt"
FANBOY_ANNOYANCE_PATH = "fanboy-annoyance.txt"

def initialize_adblock_rules():
    """Load and compile Adblock rules."""
    rules = []
    for filter_file in [EASYLIST_PATH, FANBOY_ANNOYANCE_PATH]:
        if os.path.exists(filter_file):
            with open(filter_file, "r", encoding="utf-8") as f:
                rules.extend(f.readlines())
        else:
            print(f"Adblock rules file not found: {filter_file}")
    return AdblockRules(rules)

# Initialize Adblock rules
adblock_rules = initialize_adblock_rules()

def filter_ads(content, url):
    """Filter ads and trackers using AdblockRules."""
    filtered_content = []
    try:
        for element in content.splitlines():
            if not adblock_rules.should_block(element, options={"domain": url}):
                filtered_content.append(element)
        print(f"Ad filtering completed for {url}.")
    except Exception as e:
        print(f"Ad filtering error: {e}")
    return "\n".join(filtered_content)

# Step 1: Train Model
def train_model(training_data_file):
    training_data = pd.read_csv(training_data_file)
    texts = training_data['text']
    labels = training_data['label']
    
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)
    
    vectorizers = {
        "TF": CountVectorizer(max_features=10000, ngram_range=(1, 3)),
        "TF-IDF": TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
    }
    classifiers = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "Multinomial Naive Bayes": MultinomialNB(),
        "SVM": SVC(probability=True, kernel="linear")
    }
    
    best_score = 0
    best_model = None
    best_vectorizer = None
    
    for vec_name, vectorizer in vectorizers.items():
        X_train_vec = vectorizer.fit_transform(X_train)
        X_val_vec = vectorizer.transform(X_val)
        for clf_name, classifier in classifiers.items():
            classifier.fit(X_train_vec, y_train)
            score = classifier.score(X_val_vec, y_val)
            print(f"{vec_name} + {clf_name} Validation Accuracy: {score:.4f}")
            if score > best_score:
                best_score = score
                best_model = classifier
                best_vectorizer = vectorizer
    
    with open(MODEL_FILE, "wb") as mf:
        pickle.dump(best_model, mf)
    with open(VECTORIZER_FILE, "wb") as vf:
        pickle.dump(best_vectorizer, vf)
    
    print(f"Best Model: {best_model} with Validation Accuracy: {best_score:.4f}")
    return best_model, best_vectorizer

# Step 2: Extract Text from WARC
def extract_content_from_each_warc(warc_directory, output_csv, limit_per_domain=5):
    data = []
    
    # Iterate through WARC files in the directory
    for root, dirs, files in os.walk(warc_directory):
        for file in files:
            if file.endswith(".warc") or file.endswith(".warc.gz"):
                warc_path = os.path.join(root, file)
                print(f"Processing WARC file: {warc_path}")
                
                selected_url = False  # Flag to ensure one URL per WARC file
                with open(warc_path, "rb") as stream:
                    for record in ArchiveIterator(stream):
                        if record.headers['WARC-Type'] == 'response' and not selected_url:
                            url = record.headers.get('WARC-Target-URI', None)
                            timestamp = record.headers.get('WARC-Date', None)
                            if not url or not timestamp:
                                continue
                            
                            try:
                                # Parse timestamp
                                timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
                                
                                # Parse HTML content and extract plain text
                                html_content = record.reader.read().decode('utf-8', errors='ignore')
                                html_tree = HTMLTree.parse(html_content)
                                plain_text = extract_plain_text(html_tree)
                                
                                # Filter ads using Adblock rules
                                filtered_text = filter_ads(plain_text, url)
                                
                                # Add the record to the data list
                                data.append({
                                    "url": url,
                                    "timestamp": timestamp,
                                    "folder_name": re.match(r"^\d{4}-\d{2}-\d{2}", os.path.basename(root)).group(),
                                    "text": filtered_text
                                })
                                
                                selected_url = True  # Ensure only one record is selected per WARC file
                            except Exception as e:
                                print(f"Error processing record: {e}")
    
    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    df.sort_values(by="timestamp", inplace=True)  # Sort data by timestamp
    df.to_csv(output_csv, index=False)
    print(f"Extracted {len(df)} records from WARC files and saved to {output_csv}")
    return df

# Step 3: Classify Extracted Text
def classify_text_with_timestamps(extracted_data):
    with open(MODEL_FILE, "rb") as mf:
        model = pickle.load(mf)
    with open(VECTORIZER_FILE, "rb") as vf:
        vectorizer = pickle.load(vf)
    
    tfidf_features = vectorizer.transform(extracted_data['text'])
    probabilities = model.predict_proba(tfidf_features)[:, 1]
    predictions = (probabilities > 0.5).astype(int)
    
    results_df = pd.DataFrame({
        "url": extracted_data["url"],
        "timestamp": extracted_data["timestamp"],
        "folder_name": extracted_data["folder_name"],
        "prediction": predictions
    })
    results_df.to_csv(CLASSIFIED_RESULTS_CSV, index=False)
    print(f"Classification results saved to {CLASSIFIED_RESULTS_CSV}")
    return results_df

# Step 4: Upload to Elasticsearch
def upload_to_elasticsearch(es_host, es_username, es_password, classification_df, index_name):
    es = Elasticsearch(
        hosts=[es_host],
        basic_auth=(es_username, es_password),
        verify_certs=True
    )

    actions = [
        {
            "_index": index_name,
            "_source": {
                "url": row["url"],
                "prediction": int(row["prediction"]),
                "timestamp": row["timestamp"].isoformat(),
                "folder_name": row["folder_name"]
            }
        }
        for _, row in classification_df.iterrows()
    ]

    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name)
        print(f"Index '{index_name}' created in Elasticsearch.")

    helpers.bulk(es, actions)
    print(f"Classification results uploaded to Elasticsearch index '{index_name}'.")

# Main Workflow
if __name__ == "__main__":
    print("Training model...")
    train_model(TRAINING_DATA)

    print("Extracting text and timestamps from WARC files...")
    extracted_data = extract_content_from_each_warc(WARC_DIRECTORY, EXTRACTED_CONTENT_CSV)
    
    print("Classifying extracted text...")
    classified_results = classify_text_with_timestamps(extracted_data)
    
    print("Uploading classification results to Elasticsearch...")
    upload_to_elasticsearch(ES_HOST, ES_USERNAME, ES_PASSWORD, classified_results, ES_INDEX)

sc.stop()
