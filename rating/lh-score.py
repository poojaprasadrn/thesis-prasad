import subprocess
import json
import pandas as pd
import time
import os
from concurrent.futures import ThreadPoolExecutor
import requests

# Paths
input_csv_file = 'screenshot_urls.csv'  # Input CSV with URLs & Screenshot Paths
output_directory = 'lighthouse_scores'
combined_csv_file = os.path.join(output_directory, 'combined_lighthouse_scores.csv')
failed_urls_file = os.path.join(output_directory, 'failed_urls.csv')

# Ensure output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Function to run Lighthouse & get score
def get_lighthouse_score(url, screenshot_path, retries=3, delay=10):
    """ Runs Lighthouse audit & returns SEO score """
    print(f"Fetching Lighthouse score for URL: {url}")
    
    attempt = 0
    while attempt < retries:
        try:
            # Run Lighthouse
            command = [
                'lighthouse',
                url,
                '--only-categories=seo',
                '--output=json',
                '--chrome-flags=--no-sandbox --disable-dev-shm-usage --disable-gpu --disable-extensions --disable-software-rasterizer',
                '--timeout=500000',  # Increased timeout (240s)
                '--port=9222'
            ]


            result = subprocess.run(command, capture_output=True, text=True, shell=False)

            if result.returncode == 0:
                lighthouse_json = json.loads(result.stdout)
                seo_score = lighthouse_json.get('categories', {}).get('seo', {}).get('score', 0) * 100
                return {"URL": url, "Screenshot Path": screenshot_path, "Lighthouse Score": seo_score}
            else:
                print(f"Error for {url}: {result.stderr}")

        except Exception as e:
            print(f"Exception for {url}: {e}")

        attempt += 1
        print(f"Retrying {url} (attempt {attempt}/{retries}) in {delay} seconds...")
        time.sleep(delay)

    print(f"Failed after {retries} attempts: {url}")
    return {"URL": url, "Screenshot Path": screenshot_path, "Lighthouse Score": None}

# Function to validate URLs
def validate_url(url):
    """ Checks if URL is reachable before processing """
    try:
        response = requests.head(url, timeout=10000)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

# Read CSV File
df = pd.read_csv(input_csv_file)
urls_to_process = df['URL'].tolist()
screenshots_to_process = df['Screenshot Path'].tolist()  # Assuming 'Screenshot Path' column exists

print(f"Loaded {len(urls_to_process)} URLs from CSV.")

# Filter valid URLs
valid_urls = [(url, screenshots_to_process[i]) for i, url in enumerate(urls_to_process) if validate_url(url)]
print(f"Found {len(valid_urls)} valid URLs for processing.")

combined_results = []
failed_results = []

# Run Lighthouse in parallel
with ThreadPoolExecutor(max_workers=1) as executor:  # Reduce parallelism
    futures = {executor.submit(get_lighthouse_score, url, screenshot): (url, screenshot) for url, screenshot in valid_urls}
    
    for future in futures:
        result = future.result()
        if result["Lighthouse Score"] is not None:
            combined_results.append(result)
        else:
            failed_results.append(result)

# Save results
df_results = pd.DataFrame(combined_results)
df_results.to_csv(combined_csv_file, index=False)
print(f"✅ Lighthouse scores saved: {combined_csv_file}")

# Save failed URLs separately for debugging
if failed_results:
    df_failed = pd.DataFrame(failed_results)
    df_failed.to_csv(failed_urls_file, index=False)
    print(f"⚠️ Failed URLs saved: {failed_urls_file}")
