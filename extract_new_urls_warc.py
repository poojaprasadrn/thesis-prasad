import os
import re
import tarfile
import random
from urllib.parse import urlparse
import pandas as pd
from bs4 import BeautifulSoup
from boilerpy3 import extractors

BASE_DIRECTORY = "/mnt/ceph/storage/data-in-progress/data-research/web-search/affiliate-serp-crawls"
SCREENSHOT_OUTPUT_DIR = "screenshots-300"
OUTPUT_CSV = "extracted_new_urls_with_text.csv"
MAX_URLS = 300

# Load exclusion URLs
excluded_urls = pd.read_csv("rating/screenshot_urls_final_2.csv")["URL"].dropna().str.strip().str.lower().tolist()
excluded_set = set(excluded_urls)

# BoilerPy extractor
extractor = extractors.ArticleExtractor()

# Make screenshot directory
os.makedirs(SCREENSHOT_OUTPUT_DIR, exist_ok=True)

def extract_url_from_html(html_str):
    try:
        soup = BeautifulSoup(html_str, 'html.parser')
        for tag in [
            ('meta', {'property': 'og:url'}),
            ('link', {'rel': 'canonical'}),
            ('base', {'href': True})
        ]:
            found = soup.find(*tag)
            if found and found.get('content' if 'content' in found.attrs else 'href'):
                return found.get('content' if 'content' in found.attrs else 'href')
    except Exception as e:
        print(f"❌ Error extracting URL: {e}")
    return None

def extract_domain(url):
    try:
        parsed = urlparse(url)
        return parsed.netloc.lower().replace("www.", "")
    except:
        return None

def extract_text_boilerpy(html_str):
    try:
        return ' '.join(extractor.get_content(html_str).strip().split())
    except Exception as e:
        print(f"❌ Error extracting text: {e}")
        return ""

# Collect all .tar files
tar_files = []
for root, _, files in os.walk(BASE_DIRECTORY):
    for file in files:
        if file.endswith(".tar"):
            tar_files.append(os.path.join(root, file))

print(f"✅ Found {len(tar_files)} TAR files.")

matched_data = []

while len(matched_data) < MAX_URLS and tar_files:
    tar_path = random.choice(tar_files)
    tar_files.remove(tar_path)
    print(f"\n📦 Processing TAR: {tar_path}")

    try:
        with tarfile.open(tar_path, "r") as tar:
            html_files = [m for m in tar.getmembers() if m.name.endswith(".html") and 'hit-' in m.name]
            screenshot_files = [m for m in tar.getmembers() if m.name.endswith("screenshot.png") and 'snapshot' in m.name]

            if not html_files or not screenshot_files:
                print("⚠️ Skipping (no html or screenshot)")
                continue

            html_member = html_files[0]
            screenshot_member = screenshot_files[0]

            html_content = tar.extractfile(html_member).read().decode('utf-8', errors='ignore')
            url = extract_url_from_html(html_content)
            if not url:
                print("⚠️ No URL found in HTML")
                continue

            url_clean = url.strip().lower()
            if url_clean in excluded_set:
                print("⚠️ Skipped (already in ground truth)")
                continue

            domain = extract_domain(url_clean)
            text_clean = extract_text_boilerpy(html_content)

            if not text_clean or len(text_clean.split()) < 30:
                print("⚠️ Not enough text extracted")
                continue

            screenshot_path = os.path.join(SCREENSHOT_OUTPUT_DIR, f"screenshot_{len(matched_data)}.png")
            with tar.extractfile(screenshot_member) as src, open(screenshot_path, "wb") as dst:
                dst.write(src.read())

            matched_data.append({
                "Screenshot Path": screenshot_path,
                "URL": url,
                "TAR File Path": tar_path,
                "Query Path": html_member.name,
                "Domain": domain,
                "Text": text_clean
            })
            print(f"✅ Added: {url}")
    except Exception as e:
        print(f"❌ Failed to process TAR: {e}")

    print(f"🔢 Collected: {len(matched_data)} / {MAX_URLS}")

# Save CSV
df = pd.DataFrame(matched_data)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Final data saved to: {OUTPUT_CSV}")
