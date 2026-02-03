import csv
import os
import tarfile
from resiliparse.parse.html import HTMLTree
from resiliparse.extract.html2text import extract_plain_text

input_csv_file = 'screenshot_urls_final_2.csv'
output_csv_file = 'extracted_all_urls_v2.csv'

from boilerpy3 import extractors

extractor = extractors.ArticleExtractor()

def extract_main_content_clean(html_content: bytes) -> str:
    try:
        html_str = html_content.decode('utf-8', errors='ignore')
        text = extractor.get_content(html_str)
        return ' '.join(text.strip().split())
    except Exception as e:
        print(f"Error extracting with BoilerPy3: {e}")
        return ''


def process_entry(url, tar_path, internal_path):
    if not os.path.isfile(tar_path):
        print(f"TAR file not found: {tar_path}")
        return url, ''

    try:
        with tarfile.open(tar_path, 'r') as tar:
            try:
                member = tar.getmember(internal_path)
                extracted = tar.extractfile(member)
                if extracted is None:
                    print(f"Could not extract {internal_path} from {tar_path}")
                    return url, ''
                html_content = extracted.read()
            except KeyError:
                print(f"{internal_path} not found in {tar_path}")
                return url, ''
    except Exception as e:
        print(f"Error opening TAR {tar_path}: {e}")
        return url, ''

    text = extract_main_content_clean(html_content)
    if text:
        print(f"✅ Extracted for {url} ({len(text.split())} words)")
    else:
        print(f"⚠️ Empty content for {url}")
    return url, text

def main():
    results = []

    with open(input_csv_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for idx, row in enumerate(reader):
            url = row['URL'].strip()
            tar_path = row['TAR File Path'].strip()
            internal_path = row['Query Path'].strip()

            print(f"[{idx + 1}] Processing: {url}")
            url, extracted_text = process_entry(url, tar_path, internal_path)
            results.append({
                'URL': url,
                'text': extracted_text
            })

    with open(output_csv_file, 'w', newline='', encoding='utf-8') as outfile:
        fieldnames = ['URL', 'text']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\n🎉 Done! Extracted {len(results)} URLs. Output saved to: {output_csv_file}")

if __name__ == "__main__":
    main()



# import os
# import tarfile
# import pandas as pd
# import spacy
# import re
# from bs4 import BeautifulSoup

# # Load spaCy Model
# print("Loading spaCy model...")
# nlp = spacy.load("en_core_web_sm")
# nlp.max_length = 10000000

# # Input ground truth CSV
# GROUND_TRUTH_FILE = "screenshot_urls_final_2.csv"
# OUTPUT_FILE = "warc_extracted_content_300.csv"
# MISSING_FILE = "missing_urls.csv"

# # Preprocessing and POS tagging
# def preprocess_text(text):
#     text = re.sub(r'\s+', ' ', text.strip().lower())
#     text = re.sub(r'[^\w\s]', '', text)
#     return text

# def extract_pos_tags(text):
#     doc = nlp(text)
#     return " ".join([token.pos_ for token in doc if token.is_alpha])

# # Load ground truth
# print(f"Reading ground truth from {GROUND_TRUTH_FILE}...")
# df = pd.read_csv(GROUND_TRUTH_FILE)
# results = []
# missing_urls = []

# # Iterate through ground truth entries
# print(f"Processing {len(df)} entries...")
# for idx, row in df.iterrows():
#     url = row['URL']
#     tar_path = row['TAR File Path']
#     query_path = row['Query Path']
#     screenshot = row['Screenshot Path']

#     print(f"\n[{idx+1}/{len(df)}] Processing URL: {url}")
#     if not (os.path.exists(tar_path) and tarfile.is_tarfile(tar_path)):
#         print(f"❌ File missing or not a valid tar: {tar_path}")
#         missing_urls.append(url)
#         continue

#     try:
#         with tarfile.open(tar_path, 'r') as tar:
#             try:
#                 print(f"✓ Opened TAR: {tar_path}")
#                 member = tar.getmember(query_path)
#                 print(f"✓ Found file in TAR: {query_path}")
#                 f = tar.extractfile(member)
#                 if f:
#                     print(f"✓ Extracting text and POS tags for {url}")
#                     soup = BeautifulSoup(f.read(), 'html.parser')
#                     text = preprocess_text(soup.get_text())
#                     pos_tags = extract_pos_tags(text)
#                     results.append({
#                         'screenshot': os.path.basename(screenshot),
#                         'url': url,
#                         'text': text,
#                         'pos_tags': pos_tags
#                     })
#                 else:
#                     print(f"⚠️ Could not extract file: {query_path}")
#                     missing_urls.append(url)
#             except KeyError:
#                 print(f"❌ File not found in TAR: {query_path}")
#                 missing_urls.append(url)
#     except Exception as e:
#         print(f"❌ Error opening TAR file {tar_path}: {e}")
#         missing_urls.append(url)

# # Save results
# print(f"\nSaving extracted data to {OUTPUT_FILE}...")
# df_results = pd.DataFrame(results)
# df_results.to_csv(OUTPUT_FILE, index=False)

# print(f"Saving missing URLs to {MISSING_FILE}...")
# df_missing = pd.DataFrame(missing_urls, columns=['Missing_URL'])
# df_missing.to_csv(MISSING_FILE, index=False)

# print("✅ Extraction complete!")

# # Optional: show result in notebook interface
# import ace_tools as tools; tools.display_dataframe_to_user(name="Extracted POS Tags", dataframe=df_results)


# import os
# import pandas as pd
# import spacy
# import re
# from bs4 import BeautifulSoup
# from fastwarc import ArchiveIterator
# from datetime import datetime


# # Load spaCy Model for POS tagging
# nlp = spacy.load("en_core_web_sm")
# nlp.max_length = 10000000

# # Constants
# WARC_DIRECTORY = "/mnt/ceph/storage/data-in-progress/data-research/web-search/affiliate-serp-crawls"
# GROUND_TRUTH_FILE = "screenshot_urls_final_2.csv"  # Ground truth file with URLs
# EXTRACTED_CONTENT_CSV = "warc_extracted_content_300.csv"  # Output CSV for extracted content

# # URL normalization for consistency
# def normalize_url(url):
#     if not isinstance(url, str):
#         return ""
#     return url.strip().rstrip('/').replace("http://", "https://")

# # Function to preprocess and clean text
# def preprocess_text(text):
#     if pd.isna(text) or not isinstance(text, str):
#         return ""
#     text = re.sub(r'\s+', ' ', text.strip().lower())  # Remove extra spaces and convert to lowercase
#     text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
#     return text

# def extract_pos_tags_with_check(text, url):
#     try:
#         doc = nlp(text)
#         pos_tags = [token.pos_ for token in doc if token.is_alpha]  # Extract POS tags for alphabetic tokens
#         if not pos_tags:  # Check if no POS tags were extracted
#             print(f"Warning: No POS tags extracted for URL: {url}")
#             return None  # Return None if no POS tags are found
#         return " ".join(pos_tags)  # Return POS tags as a space-separated string
#     except Exception as e:
#         print(f"Error processing POS tags for URL: {url}, Error: {e}")
#         return None  # Return None if there's any error during processing

# # Function to extract text from WARC for only the ground truth URLs
# def extract_text_from_warc(warc_directory, urls_to_extract):
#     extracted_data = []
#     missing_pos_data = []  # List to store URLs with missing POS tags
    
#     # Convert the URLs list to a set for faster lookup
#     urls_to_extract_set = set(urls_to_extract)
    
#     # Iterate through WARC files in the directory
#     for root, dirs, files in os.walk(warc_directory):
#         for file in files:
#             if file.endswith(".warc") or file.endswith(".warc.gz") or file.endswith(".tar"):
#                 warc_path = os.path.join(root, file)
#                 print(f"Processing WARC file: {warc_path}")
                
#                 try:
#                     with open(warc_path, "rb") as stream:
#                         for record in ArchiveIterator(stream):
#                             if record.headers.get('WARC-Type') != 'response':
#                                 continue

#                             url = normalize_url(record.headers.get('WARC-Target-URI', ''))
#                             timestamp = record.headers.get('WARC-Date', None)
#                             if not url or url not in urls_to_extract_set or not timestamp:
#                                 continue

#                             try:
#                                 timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
#                                 html_bytes = record.reader.read()
#                                 if not html_bytes:
#                                     continue

#                                 html_content = html_bytes.decode('utf-8', errors='ignore')
#                                 soup = BeautifulSoup(html_content, 'html.parser')
#                                 text = soup.get_text()
#                                 processed_text = preprocess_text(text)
#                                 pos_tags = extract_pos_tags_with_check(processed_text, url)

#                                 if not pos_tags:
#                                     missing_pos_data.append(url)

#                                 extracted_data.append({
#                                     "url": url,
#                                     "screenshot": os.path.basename(screenshot),
#                                     "text": processed_text,
#                                     "pos_tags": pos_tags
#                                 })
#                             except Exception as e:
#                                 print(f"❌ Error reading record for {url}: {e}")
#                 except Exception as e:
#                     print(f"❌ Failed to open WARC file {warc_path}: {e}")

    
#     # Convert to DataFrame and save
#     df = pd.DataFrame(extracted_data)
#     df.sort_values(by="timestamp", inplace=True)  # Sort data by timestamp
#     df = df.drop_duplicates(subset="url", keep="first")  # Keep only one entry per URL
#     df.to_csv(EXTRACTED_CONTENT_CSV, index=False)
#     print(f"Extracted {len(df)} records from WARC files and saved to {EXTRACTED_CONTENT_CSV}")
    
#     # Save missing POS tags URLs to a separate file for analysis
#     missing_pos_df = pd.DataFrame(missing_pos_data, columns=["url"])
#     missing_pos_df.to_csv("missing_pos_tags_urls.csv", index=False)
#     print(f"URLs with missing POS tags saved to 'missing_pos_tags_urls.csv'")
    
#     return df

# # Load ground truth file with URLs
# ground_truth = pd.read_csv(GROUND_TRUTH_FILE)

# # Get URLs from the ground truth data
# urls_to_extract = ground_truth['URL'].tolist()
# print(f"Extracting data for {len(urls_to_extract)} URLs...")

# # Extract text for these URLs from the WARC files in the specified directory
# extracted_data = extract_text_from_warc(WARC_DIRECTORY, urls_to_extract)

# # Optionally, you can print or analyze the extracted data
# print(f"Extracted data for {len(extracted_data)} URLs.")
