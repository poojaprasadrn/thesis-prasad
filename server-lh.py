import os
import json
import subprocess
import pandas as pd
from fastwarc import ArchiveIterator
import requests
import time

def get_lighthouse_score(url):
    print(f"Fetching Lighthouse score for URL: {url}")
    try:
        response = requests.head(url, timeout=5)
        if response.status_code != 200:
            print(f"URL is not accessible, status code: {response.status_code}")
            return None

        command = [
            'lighthouse',
            url,
            '--only-categories=seo',
            '--output=json',
            '--chrome-flags=--headless --no-sandbox --disable-dev-shm-usage',
            '--timeout=120000',
            '--port=9222'
        ]
        result = subprocess.run(command, capture_output=True, text=True, shell=False)

        if result.returncode == 0:
            lighthouse_json = json.loads(result.stdout)
            seo_score = lighthouse_json.get('categories', {}).get('seo', {}).get('score', 0) * 100
            return seo_score
        else:
            print(f"Error for {url}: {result.stderr}")
    except Exception as e:
        print(f"Exception for {url}: {e}")
    return None

def fetch_urls_from_warc(warc_file_path):
    urls = []
    try:
        with open(warc_file_path, 'rb') as stream:
            for record in ArchiveIterator(stream):
                if record.headers['WARC-Type'] == 'response':
                    content_type = record.http_headers.get('Content-Type', '')
                    if 'text/html' in content_type:
                        url = record.headers['WARC-Target-URI']
                        if url.startswith("http://") or url.startswith("https://"):
                            urls.append(url)
    except Exception as e:
        print(f"Error reading WARC file {warc_file_path}: {e}")
    return urls

warc_directory = '/mnt/ceph/storage/data-in-progress/data-research/web-search/affiliate-serp-crawls'
urls_file_path = "/home/yili5634/Desktop/thesis-pooja/warc_extracted_5_urls.csv"
output_path = "predicted_lh_scores.csv"

urls_df = pd.read_csv(urls_file_path)
urls_df.columns = urls_df.columns.str.lower().str.strip()
urls_to_check = urls_df["url"].tolist()

warc_files = [os.path.join(warc_directory, f) for f in os.listdir(warc_directory) if f.endswith('.warc') or f.endswith('.warc.gz')]
all_urls = set()
for warc_file in warc_files:
    all_urls.update(fetch_urls_from_warc(warc_file))

scores = {}

chrome_process = subprocess.Popen([
    'google-chrome', '--headless', '--disable-gpu', '--remote-debugging-port=9222', '--disable-software-rasterizer'
])

time.sleep(5)

for url in urls_to_check:
    if url in all_urls:
        scores[url] = get_lighthouse_score(url)

chrome_process.terminate()
chrome_process.wait()

data = [{"url": url, "lighthouse score": int(score) if score is not None else None} for url, score in scores.items()]
df = pd.DataFrame(data)
df.to_csv(output_path, index=False)
print(f"Predicted scores saved to {output_path}")


# import os
# import json
# import subprocess
# import pandas as pd
# from fastwarc import ArchiveIterator
# from http.server import SimpleHTTPRequestHandler, HTTPServer
# import threading
# from pyspark import SparkConf, SparkContext
# import time
# import requests
# from concurrent.futures import ThreadPoolExecutor
# import gc
# import psutil
# import datetime

# # Function to log memory usage
# def log_memory_usage(context="Unknown"):
#     process = psutil.Process(os.getpid())
#     memory_info = process.memory_info()
#     print(f"[{datetime.datetime.now()}] [{context}] Memory Usage: RSS={memory_info.rss / (1024 * 1024)} MB, VMS={memory_info.vms / (1024 * 1024)} MB")

# # Spark configuration
# conf = (
#     SparkConf()
#     .setMaster('k8s://https://k8s.srv.webis.de')  # Kubernetes master URL
#     #.setMaster('local[*]')
#     .setAppName('LighthouseScoreCalculation')     # Application name
#     .set('spark.executor.instances', 2)           # 2 executors to reduce memory pressure
#     .set('spark.executor.memory', '10g')          # Memory per executor adjusted to fit within limits
#     .set('spark.executor.cores', '1')             # Use 1 core per executor to reduce CPU load
#     .set('spark.kubernetes.executor.request.cores', '1') # Request 1 core per executor
#     .set('spark.kubernetes.executor.limit.cores', '1')       # Set cores per executor to 1 to match the CPU limit
#     .set('spark.driver.memory', '8g')             # Reduce driver memory to 8GB
#     .set('spark.kubernetes.container.image', 'registry.webis.de/code-teaching/theses/thesis-prasad:v1.0')  # Docker image
#     .set('spark.kubernetes.namespace', 'spark-jobs')  # Kubernetes namespace
#     .set('spark.kubernetes.authenticate.driver.serviceAccountName', 'spark')  # Service account
#     .set('spark.kubernetes.container.image.pullSecrets', 'webis-registry-credentials')  # Image pull secret
#     .set('spark.kubernetes.executor.podCreationTimeout', '600s')  # Increase executor pod creation timeout
#     .set("spark.kubernetes.driver.annotation.sidecar.istio.io/inject", "false")
#     .set("spark.kubernetes.executor.annotation.sidecar.istio.io/inject", "false")
#     .set('spark.kubernetes.driver.volumes.hostPath.storage.options.path', '/mnt/ceph/storage')
#     .set('spark.kubernetes.driver.volumes.hostPath.storage.mount.path', '/mnt/ceph/storage')
#     .set('spark.kubernetes.executor.volumes.hostPath.storage.options.path', '/mnt/ceph/storage')
#     .set('spark.kubernetes.executor.volumes.hostPath.storage.mount.path', '/mnt/ceph/storage')
# )

# # Initialize Spark context
# sc = SparkContext(conf=conf)
# log_memory_usage("After SparkContext Initialization")

# # Function to fetch valid URLs from a WARC file
# def fetch_urls_from_warc(warc_file_path):
#     urls = []
#     try:
#         with open(warc_file_path, 'rb') as stream:
#             for record in ArchiveIterator(stream):
#                 if record.headers['WARC-Type'] == 'response':
#                     content_type = record.http_headers.get('Content-Type', '')
#                     print(f"Processing record with Content-Type: {content_type}")
#                     if 'text/html' in content_type:
#                         url = record.headers['WARC-Target-URI']
#                         if url.startswith("http://") or url.startswith("https://"):
#                             urls.append(url)
#                             print(f"Extracted URL: {url}")
#                     else:
#                         print(f"Skipped non-HTML content type: {content_type}")
#         if not urls:
#             print(f"No valid URLs found in {warc_file_path}")
#     except Exception as e:
#         print(f"Error reading WARC file {warc_file_path}: {e}")

#     return urls

# # function to get Lighthouse score for a given URL
# def get_lighthouse_score(url):
#     print(f"Fetching Lighthouse score for URL: {url}")
#     log_memory_usage("Before Fetching Lighthouse Score")
#     try:
#         # Check if the URL is accessible
#         response = requests.head(url, timeout=5)
#         if response.status_code != 200:
#             print(f"URL is not accessible, received status code: {response.status_code}")
#             return None

#         command = [
#             'lighthouse',
#             url,
#             '--only-categories=seo',  # Limit the audit to only SEO category 
#             '--output=json',
#             '--chrome-flags=--headless --no-sandbox --disable-dev-shm-usage',
#             '--timeout=120000',  # Reduce the timeout to 120 seconds
#             '--port=9222'  # Connect to the running Chrome instance
#         ]

#         print(f"Running command: {' '.join(command)}")
#         result = subprocess.run(command, capture_output=True, text=True, shell=False)

#         if result.returncode == 0:
#             # Parse the JSON output from Lighthouse
#             lighthouse_json = json.loads(result.stdout)
#             seo_score = lighthouse_json.get('categories', {}).get('seo', {}).get('score', 0) * 100
#             log_memory_usage("After Fetching Lighthouse Score")
#             return seo_score
#         else:
#             print(f"Error for {url}: {result.stderr}")

#         print(f"Failed to get Lighthouse score for {url}.")
#         return None

#     except Exception as e:
#         error_message = f"Exception for {url}: {e}"
#         print(error_message)
#         return None

# # Start Chrome programmatically
# chrome_process = subprocess.Popen([
#     'google-chrome',
#     '--headless',
#     '--disable-gpu',
#     '--remote-debugging-port=9222',
#     '--disable-software-rasterizer'
# ])
# log_memory_usage("After Starting Chrome")

# # Directory path with WARC files
# warc_directory = '/mnt/ceph/storage/data-in-progress/data-research/web-search/affiliate-serp-crawls'


# def find_warc_files(directory):
#     warc_files = []
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file.endswith('.warc') or file.endswith('.warc.gz'):
#                 warc_files.append(os.path.join(root, file))
#     return warc_files
    

# # Define the output path in the mounted Ceph directory
# output_directory = '/mnt/ceph/storage/data-tmp/current/yili5634/lighthouse_scores'
# combined_csv_file = os.path.join(output_directory, 'combined_lighthouse_scores.csv')
# if not os.path.exists(output_directory):
#     os.makedirs(output_directory)

# # Debugging: Check if the directory exists and list its contents
# if not os.path.exists(warc_directory):
#     print(f"Directory does not exist: {warc_directory}")
#     warc_files = []
# else:
#     print(f"Directory exists: {warc_directory}")
#     all_files = os.listdir(warc_directory)
#     print(f"Files found in directory: {all_files}")
#     warc_files = find_warc_files(warc_directory)

# log_memory_usage("After Finding WARC Files")

# # Proceed with processing all WARC files if available
# if not warc_files:
#     print("No WARC files found in the directory.")
# else:
#     urls_to_process = []
#     for warc_file in warc_files:
#         urls = fetch_urls_from_warc(warc_file)
#         urls_to_process.extend(urls)

#     # Limit to 200 URLs for processing, excluding those with non-200 status code
#     urls_to_process = urls_to_process[:2000]  # Limit to 200 URLs initially
#     valid_urls = []
#     start_time = datetime.datetime.now()
#     for url in urls_to_process:
#         try:
#             response = requests.head(url, timeout=5)
#             if response.status_code == 200:
#                 valid_urls.append(url)
#             if len(valid_urls) >= 500:
#                 break
#         except requests.exceptions.RequestException as e:
#             print(f"Skipping URL due to error: {e}")

#     end_time = datetime.datetime.now()
#     duration = end_time - start_time
#     print(f"Time taken to validate URLs: {duration}")

#     log_memory_usage("After Validating URLs")

#     if valid_urls:
#         # Set up a simple HTTP server to serve the HTML files locally
#         PORT = 8080
#         handler = SimpleHTTPRequestHandler
#         httpd = HTTPServer(('localhost', PORT), handler)
#         server_thread = threading.Thread(target=httpd.serve_forever)
#         server_thread.daemon = True
#         server_thread.start()
#         print(f"Serving HTTP on port {PORT}...")

#         combined_results = []

#         # Using ThreadPoolExecutor for parallel execution
#         start_processing_time = datetime.datetime.now()
#         with ThreadPoolExecutor(max_workers=4) as executor:  # max_workers to 4 for better parallelism and memory usage
#             futures = [executor.submit(get_lighthouse_score, url) for url in valid_urls]
#             for future, url in zip(futures, valid_urls):
#                 score = future.result()
#                 if score is not None:
#                     result = {'URL': url, 'Lighthouse Score': score}
#                     combined_results.append(result)

#                     # Append each result to the combined CSV file
#                     df = pd.DataFrame([result])
#                     df.to_csv(combined_csv_file, mode='a', header=not os.path.exists(combined_csv_file), index=False)
#                     print(f"Appended result to {combined_csv_file}.")

#                 # Clear memory after processing each URL
#                 gc.collect()
#         end_processing_time = datetime.datetime.now()
#         processing_duration = end_processing_time - start_processing_time
#         print(f"Time taken to process URLs: {processing_duration}")

#         log_memory_usage("After Processing URLs")

#         # Shutdown the HTTP server
#         httpd.shutdown()
#         server_thread.join()

#     else:
#         print("No valid URLs could be processed.")

# # Stop the Chrome process
# chrome_process.terminate()
# chrome_process.wait()  # Ensure Chrome process is fully terminated

# # Clear any remaining processes related to Chrome
# for proc in psutil.process_iter(['pid', 'name']):
#     if 'chrome' in proc.info['name'].lower():
#         os.kill(proc.info['pid'], 9)

# # Clear memory and collect garbage before starting a new process
# gc.collect()
# log_memory_usage("After Garbage Collection")

# # Stop the SparkContext when done
# sc.stop()
# log_memory_usage("After Stopping SparkContext")



