from datetime import datetime
import cv2
import psutil
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re
from fastwarc import ArchiveIterator
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import subprocess
import json
from http.server import SimpleHTTPRequestHandler, HTTPServer
import threading
from elasticsearch import Elasticsearch, helpers

# Configuring the Spark context
conf = (
    SparkConf()
    .setMaster('local[*]')
    .setAppName('ExtractFeatures')
    .set('spark.executor.memory', '13g')
    .set('spark.executor.cores', '2')
    .set('spark.kubernetes.container.image', 'registry.webis.de/code-teaching/theses/thesis-prasad:v1.0')
    .set('spark.kubernetes.namespace', 'spark-jobs')
)
sc = SparkContext(conf=conf)
spark = SparkSession.builder.config(conf=conf).getOrCreate()

# Directory path with WARC files
warc_directory = '/mnt/ceph/storage/data-in-progress/data-research/web-search/affiliate-serp-crawls'

def compute_feature_importance():
    seo_features_path = '/mnt/ceph/storage/data-tmp/current/yili5634/seo_features_all_urls.csv'
    if os.path.exists(seo_features_path):
        print("SEO features file found. Computing feature importance...")
        seo_features_pd = pd.read_csv(seo_features_path)
        feature_columns = [
            'meta_description', 'meta_keywords', 'title', 'h1_tags', 'h2_tags',
            'alt_text_count', 'internal_links', 'external_links',
            'keyword_density', 'canonical_tag_present', 'robots_txt_present',
            'sitemap_present', 'ads_present', 'tools_used', 'url_length',
            'social_media_links', 'caching_tools'
        ]
        target_column = 'lighthouse_score'

        X = seo_features_pd[feature_columns]
        y = seo_features_pd[target_column]
      
        
        svr_model = SVR(kernel='linear')


        
        svr_model.fit(X, y)

        feature_importances = np.abs(svr_model.coef_[0])

        feature_importance_df = pd.DataFrame({'Feature': feature_columns, 'Importance': feature_importances})
        feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
        feature_importance_output_path = '/mnt/ceph/storage/data-tmp/current/yili5634/feature_importance_all_urls.csv'
        feature_importance_df.to_csv(feature_importance_output_path, index=False)
        print(f"Feature importance saved to: {feature_importance_output_path}")

        es = Elasticsearch(
            hosts=["https://elasticsearch.srv.webis.de/"],
            basic_auth=('yili5634', 'KqAk50zmz80BSjtn'),
            verify_certs=True
        )

        # Upload SEO features to Elasticsearch
        seo_features_data = seo_features_pd.to_dict(orient='records')
        seo_features_index = 'wstud_yili5634_seo_features'
        seo_features_actions = [
            {"_index": seo_features_index, "_source": feature}
            for feature in seo_features_data
        ]
        if not es.indices.exists(index=seo_features_index):
            es.indices.create(index=seo_features_index)
        helpers.bulk(es, seo_features_actions)
        print(f"SEO features uploaded to Elasticsearch index '{seo_features_index}'")

        # Upload Feature Importance to Elasticsearch
        feature_importance_data = feature_importance_df.to_dict(orient='records')
        feature_importance_index = 'wstud_yili5634_feature_importance'
        feature_importance_actions = [
            {"_index": feature_importance_index, "_source": feature}
            for feature in feature_importance_data
        ]
        if not es.indices.exists(index=feature_importance_index):
            es.indices.create(index=feature_importance_index)
        helpers.bulk(es, feature_importance_actions)
        print(f"Feature importance uploaded to Elasticsearch index '{feature_importance_index}'")

def extract_seo_features(record):
    url = record['url']
    timestamp = record['timestamp']
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')

            meta_description = len(soup.find('meta', attrs={'name': 'description'})['content']) if soup.find('meta', attrs={'name': 'description'}) and 'content' in soup.find('meta', attrs={'name': 'description'}).attrs else 0
            meta_keywords = len(soup.find('meta', attrs={'name': 'keywords'})['content']) if soup.find('meta', attrs={'name': 'keywords'}) and 'content' in soup.find('meta', attrs={'name': 'keywords'}).attrs else 0
            title_tag = len(soup.title.string) if soup.title and soup.title.string else 0
            h1_tags = len(soup.find_all('h1'))
            h2_tags = len(soup.find_all('h2'))
            alt_text_count = len([img for img in soup.find_all('img') if img.has_attr('alt')])
            internal_links = len([link for link in soup.find_all('a', href=True) if urlparse(link['href']).netloc in ("", urlparse(url).netloc)])
            external_links = len([link for link in soup.find_all('a', href=True) if urlparse(link['href']).netloc not in ("", urlparse(url).netloc)])

            # Dynamic Keyword Extraction using TF-IDF
            all_text = soup.get_text().lower()
            if not all_text.strip():
                keyword_density = 0
            else:
                # Apply TF-IDF only if there's meaningful content
                vectorizer = TfidfVectorizer(stop_words='english', max_features=10)
                try:
                    tfidf_matrix = vectorizer.fit_transform([all_text])
                    keywords = vectorizer.get_feature_names_out()
                    keyword_density = sum([all_text.count(keyword) for keyword in keywords]) / len(all_text.split()) if all_text else 0
                except ValueError:
                    keyword_density = 0 
     

            canonical_tag_present = len(soup.find('link', rel='canonical')['href']) if soup.find('link', rel='canonical') and 'href' in soup.find('link', rel='canonical').attrs else 0
            robots_txt_present = len(requests.get(f"{urlparse(url).scheme}://{urlparse(url).netloc}/robots.txt", timeout=5).text) if requests.get(f"{urlparse(url).scheme}://{urlparse(url).netloc}/robots.txt", timeout=5).status_code == 200 else 0
            sitemap_present = len(soup.find('link', rel='sitemap')['href']) if soup.find('link', rel='sitemap') and 'href' in soup.find('link', rel='sitemap').attrs else 0
            ads_present = len(soup.find_all(attrs={'class': re.compile(r'ads|ad-container')}))
            tools_used = len([script for script in soup.find_all('script', src=True) if re.search(r'google-analytics|fbq', script['src'])])
            url_length = len(url)
            social_media_links = len([link for link in soup.find_all('a', href=True) if re.search(r'facebook|twitter|instagram', link['href'])])
            caching_tools = len(soup.find('meta', attrs={'http-equiv': 'cache-control'})['content']) if soup.find('meta', attrs={'http-equiv': 'cache-control'}) and 'content' in soup.find('meta', attrs={'http-equiv': 'cache-control'}).attrs else 0
            
        
            # Get Lighthouse score
            lighthouse_score = get_lighthouse_score(url)

            return {
                'url': url,
                'meta_description': meta_description,
                'meta_keywords': meta_keywords,
                'title': title_tag,
                'h1_tags': h1_tags,
                'h2_tags': h2_tags,
                'alt_text_count': alt_text_count,
                'internal_links': internal_links,
                'external_links': external_links,
                'keyword_density': float(keyword_density),
                'canonical_tag_present': canonical_tag_present,
                'robots_txt_present': robots_txt_present,
                'sitemap_present': sitemap_present,
                'ads_present': ads_present,
                'tools_used': tools_used,
                'url_length': url_length,
                'social_media_links': social_media_links,
                'caching_tools': caching_tools,
                'lighthouse_score': lighthouse_score,
                'timestamp': timestamp.isoformat(),
            }
        else:
            return {'url': url, 'timestamp': timestamp.isoformat(), 'error': 'Failed to fetch'}
    except requests.RequestException as e:
        return {'url': url,'timestamp': timestamp.isoformat(), 'error': str(e)}
    
PORT = 8080
handler = SimpleHTTPRequestHandler
httpd = HTTPServer(('localhost', PORT), handler)
server_thread = threading.Thread(target=httpd.serve_forever)
server_thread.daemon = True
server_thread.start()
print(f"Serving HTTP on port {PORT}...")

import socket

def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # Bind to a free port
        return s.getsockname()[1]

# Function to get Lighthouse score for a given URL
def get_lighthouse_score(url):
    try:
        # Lighthouse command
        command = [
            'lighthouse',
            url,
            '--only-categories=seo',
            '--output=json',
            '--timeout=300000',
            '--chrome-flags="--headless --disable-gpu --no-sandbox --disable-dev-shm-usage --disable-storage-reset --throttling-method=provided"',
            '--verbose',
        ]
        result = subprocess.run(command, capture_output=True, text=True, shell=False)

        if result.returncode == 0:
            # Parse JSON output
            lighthouse_json = json.loads(result.stdout)
            seo_score = lighthouse_json.get('categories', {}).get('seo', {}).get('score', 0) * 100
            return seo_score
        else:
            if "NO_FCP" in result.stderr:
                print(f"NO_FCP error for {url}. Skipping.")
                return None
            print(f"Error for {url}: {result.stderr}")
            return None
    except Exception as e:
        print(f"Exception for {url}: {e}")
        return None


# Skip WARC file processing and URL extraction if SEO features already exist
seo_features_path = '/mnt/ceph/storage/data-tmp/current/yili5634/seo_features_all_urls.csv'
if os.path.exists(seo_features_path):
    print("SEO features already extracted. Skipping WARC file and URL processing.")
    compute_feature_importance()
else:
    def find_warc_files(directory):
        print('extracting warc files...')
        warc_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(('.warc', '.warc.gz')):
                    warc_files.append(os.path.join(root, file))
        return warc_files

    def fetch_urls_from_warc(warc_file_path):
        print('extracting url...')
        data = []
        try:
            with open(warc_file_path, 'rb') as stream:
                for record in ArchiveIterator(stream):
                    if record.headers['WARC-Type'] == 'response':
                        url = record.headers.get('WARC-Target-URI', None)
                        timestamp = record.headers.get('WARC-Date', None)
                        if not url or not timestamp:
                            continue
                        try:
                            timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
                            data.append({"url": url, "timestamp": timestamp})
                            break
                        except ValueError as e:
                            print(f"Error parsing timestamp for URL {url}: {e}")
        except Exception as e:
            print(f"Error reading WARC file {warc_file_path}: {e}")
        return data

    warc_files = find_warc_files(warc_directory)
    urls = []
    for warc_file in warc_files:
        urls.extend(fetch_urls_from_warc(warc_file))
        if len(urls) >= 100000:
            break
    urls = urls[:100000]

    if not urls:
        print("No URLs were extracted from the WARC files.")
    else:
        print(f"Extracted {len(urls)} URLs for processing.")

    if urls:
        urls_rdd = sc.parallelize(urls)

        def process_url(record):
            try:
                return extract_seo_features(record)
            except Exception as e:
                print(f"Error processing URL {record['url']}: {e}")
                return None

        seo_features_rdd = urls_rdd.map(process_url).filter(lambda x: x is not None)

        if not seo_features_rdd.isEmpty():
            seo_features_df = seo_features_rdd.toDF()
            seo_features_pd = seo_features_df.toPandas()
            seo_features_pd.dropna(inplace=True)

            seo_features_output_path = '/mnt/ceph/storage/data-tmp/current/yili5634/seo_features_all_urls.csv'
            seo_features_pd.to_csv(seo_features_output_path, index=False)
            print(f"All SEO features saved to: {seo_features_output_path}")
            compute_feature_importance()
        else:
            print("No valid SEO features extracted. Computing feature importance if previous data exists.")
            compute_feature_importance()

chrome_process = subprocess.Popen([
  'google-chrome',
    '--headless',
    '--disable-gpu',
    '--disable-software-rasterizer',
    '--no-sandbox',
    '--user-data-dir=/tmp/chrome-user-data'
])

sc.stop()
httpd = HTTPServer(('localhost', 8080), SimpleHTTPRequestHandler)
httpd.shutdown()
