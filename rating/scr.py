import os
import re
import tarfile
import random
from urllib.parse import urlparse
import pandas as pd
from bs4 import BeautifulSoup

BASE_DIRECTORY = "/mnt/ceph/storage/data-in-progress/data-research/web-search/affiliate-serp-crawls"
SCREENSHOT_OUTPUT_DIR = "screenshots-300"
OUTPUT_CSV = "screenshot_urls_final_2.csv"
MAX_REVIEWS = 200
MAX_NONREVIEWS = 100
MAX_SCREENSHOTS = MAX_REVIEWS + MAX_NONREVIEWS

# Load exclusion URLs from CSVs
exclude_1 = pd.read_csv("../warc_extracted_5_urls.csv")  # update path if needed
exclude_2 = pd.read_csv("ground_truth_spam_with_url.csv")
excluded_urls = pd.concat([exclude_1["url"], exclude_2["URL"]]).drop_duplicates().str.strip().str.lower()

# Review-related regex
review_url_pattern = re.compile(
    r"(best[-]?\w*|top[-]?\w*|review[s]?|rating[s]?|feedback|opinion[s]?|product[-]?(review[s]?|rating[s]?)|comparison[s]?|pros[-]?and[-]?cons)",
    re.IGNORECASE
)
exclude_domains = ["amazon", "pinterest","youtube","quora","ebay","flipkart","etsy","linkedin","outdoorgearlab","bestcraftorganizer","wahoox","play.google.com","ukcurtainpoles","onelittleproject","forum.electricunicycle","houzz","wikipedia","american-footballshop"]
seen_domains = set() 

def extract_url_from_html(html_path):
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            url = soup.find('meta', {'property': 'og:url'})
            if url: return url.get('content')
            url = soup.find('link', {'rel': 'canonical'})
            if url: return url.get('href')
            base = soup.find('base', {'href': True})
            if base: return base.get('href')
    except Exception as e:
        print(f"Error extracting URL from {html_path}: {e}")
    return None

def extract_domain(url):
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace("www.", "")
        return domain
    except:
        return None
    
# Find all TAR files
tar_files = []
for root, dirs, files in os.walk(BASE_DIRECTORY):
    for file in files:
        if file.endswith(".tar"):
            tar_files.append(os.path.join(root, file))
print(f"✅ Found {len(tar_files)} TAR files.")

matched_data = []
review_count = 0
nonreview_count = 0


while len(matched_data) < MAX_SCREENSHOTS and tar_files:
    tar_file = random.choice(tar_files)
    tar_files.remove(tar_file)
    print(f"\n🔄 Processing TAR File: {tar_file}")

    try:
        with tarfile.open(tar_file, "r") as tar:
            html_files = [m for m in tar.getmembers() if m.name.endswith(".html") and 'hit-' in m.name]
            if not html_files:
                print("⚠️ No HTML file found.")
                continue

            html_member = html_files[0]
            os.makedirs(SCREENSHOT_OUTPUT_DIR, exist_ok=True)
            html_path = os.path.join(SCREENSHOT_OUTPUT_DIR, f"html_{len(matched_data)}.html")

            with tar.extractfile(html_member) as src, open(html_path, "wb") as dst:
                dst.write(src.read())

            url = extract_url_from_html(html_path)
            if not url:
                print("⚠️ No URL found.")
                continue

            url_clean = url.strip().lower()
            domain = extract_domain(url_clean)
            if not domain:
                print("⚠️ Could not extract domain.")
                continue

            if url_clean in excluded_urls.values:
                print("⚠️ URL already exists in excluded set.")
                continue

            if any(d in url_clean for d in exclude_domains):
                print("⚠️ Skipping excluded domain.")
                continue

            if domain in seen_domains:
                print(f"⚠️ Already selected a URL from domain: {domain}")
                continue

            is_review = bool(review_url_pattern.search(url_clean))
            is_nonreview = not is_review

            if is_review and review_count >= MAX_REVIEWS:
                continue
            if is_nonreview and nonreview_count >= MAX_NONREVIEWS:
                continue

            snapshot_files = [m for m in tar.getmembers() if m.name.endswith("screenshot.png") and 'snapshot' in m.name]
            if not snapshot_files:
                print("⚠️ No screenshot found.")
                continue

            screenshot_member = snapshot_files[0]
            screenshot_path = os.path.join(SCREENSHOT_OUTPUT_DIR, f"screenshot_{len(matched_data)}.png")
            with tar.extractfile(screenshot_member) as src, open(screenshot_path, "wb") as dst:
                dst.write(src.read())

            matched_data.append({
                "Screenshot Path": screenshot_path,
                "URL": url,
                "TAR File Path": tar_file,
                "Query Path": html_member.name,
                "Domain": domain
            })
            seen_domains.add(domain)

            if is_review:
                review_count += 1
                print(f"✅ Review: {url}")
            else:
                nonreview_count += 1
                print(f"✅ Non-Review: {url}")

    except Exception as e:
        print(f"❌ Error processing TAR file: {e}")

    print(f"🔢 Total screenshots: {len(matched_data)} | Reviews: {review_count} | Non-reviews: {nonreview_count}")

# Save to CSV
df = pd.DataFrame(matched_data)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Final screenshot URLs saved to {OUTPUT_CSV}")


# import os
# import re
# import tarfile
# import random
# import pandas as pd
# from bs4 import BeautifulSoup

# BASE_DIRECTORY = "/mnt/ceph/storage/data-in-progress/data-research/web-search/affiliate-serp-crawls"
# SCREENSHOT_OUTPUT_DIR = "screenshots"  # Directory to store extracted screenshots
# OUTPUT_CSV = "screenshot_urls_final.csv"
# MAX_SCREENSHOTS = 300  # The number of unique screenshots to extract

# # Load the two exclusion CSV files
# exclude_1 = pd.read_csv("../warc_extracted_5_urls.csv")  # previously extracted screenshots
# exclude_2 = pd.read_csv("ground_truth_spam_with_url.csv")  # already labeled

# # Combine both exclusion lists
# excluded_urls = pd.concat([exclude_1["url"], exclude_2["URL"]]).drop_duplicates().reset_index(drop=True)

# # Regex pattern for matching review-related URLs
# review_url_pattern = r"(best|reviews?|ratings?|feedback|opinions?|product[-\w]*reviews?)"
# #re.compile(r'(best|reviews?|ratings?|feedback|opinions?|product[-\w]*reviews?)', re.IGNORECASE)
# non_review_exclude_domains = ["amazon", "pinterest"]

# review_urls = all_urls_df[
#     all_urls_df["URL"].str.contains(review_pattern, case=False, na=False) &
#     ~all_urls_df["URL"].str.contains("|".join(non_review_exclude_domains), case=False, na=False) &
#     ~all_urls_df["URL"].isin(excluded_urls)
# ]
# def extract_url_from_html(html_path):
#     """Extracts URL from HTML files."""
#     try:
#         with open(html_path, 'r', encoding='utf-8') as f:
#             soup = BeautifulSoup(f, 'html.parser')
#             # Try to find the canonical URL or OG URL
#             url = soup.find('meta', {'property': 'og:url'})
#             if url:
#                 return url.get('content')
#             url = soup.find('link', {'rel': 'canonical'})
#             if url:
#                 return url.get('href')
#             # If no meta or canonical tag, return the base URL from the HTML
#             return soup.find('base', {'href': True}).get('href', '#')
#     except Exception as e:
#         print(f"Error extracting URL from {html_path}: {e}")
#         return None

# # Scan directories for TAR files
# tar_files = []
# for root, dirs, files in os.walk(BASE_DIRECTORY):
#     for file in files:
#         if file.endswith(".tar"):
#             tar_files.append(os.path.join(root, file))

# print(f"✅ Found {len(tar_files)} TAR files.")

# # Step 2: Extract screenshots and URLs from TAR files
# matched_data = []
# while len(matched_data) < MAX_SCREENSHOTS and tar_files:
#     tar_file = random.choice(tar_files)  # Pick a random TAR file
#     tar_files.remove(tar_file)  # Remove from available TAR files

#     print(f"\n🔄 Processing TAR File: {tar_file}")

#     try:
#         with tarfile.open(tar_file, "r") as tar:
#             # Searching for any .html file in the hit-* folders (not inside snapshot)
#             html_files = [m for m in tar.getmembers() if m.name.endswith(".html") and 'hit-' in m.name]
#             if html_files:
#                 html_member = html_files[0]  # Pick the first HTML file found
#                 html_path = os.path.join(SCREENSHOT_OUTPUT_DIR, f"html_{len(matched_data)}.html")
#                 os.makedirs(SCREENSHOT_OUTPUT_DIR, exist_ok=True)
#                 with tar.extractfile(html_member) as src, open(html_path, "wb") as dst:
#                     dst.write(src.read())
                
#                 print(f"✅ Extracted HTML: {html_path}")
                
#                 # Extract URL from the HTML file
#                 url = extract_url_from_html(html_path)
#                 if url and review_url_pattern.search(url):
#                     print(f"✅ Extracted URL: {url}")
                    
#                     # Searching for screenshot in the snapshot folder
#                     snapshot_files = [m for m in tar.getmembers() if m.name.endswith("screenshot.png") and 'snapshot' in m.name]
#                     if snapshot_files:
#                         screenshot_member = snapshot_files[0]  # Pick the first screenshot found
#                         screenshot_path = os.path.join(SCREENSHOT_OUTPUT_DIR, f"screenshot_{len(matched_data)}.png")
#                         with tar.extractfile(screenshot_member) as src, open(screenshot_path, "wb") as dst:
#                             dst.write(src.read())
#                         print(f"✅ Extracted Screenshot: {screenshot_path}")
#                         matched_data.append({
#                             "Screenshot Path": screenshot_path,
#                             "URL": url,
#                             "TAR File Path": tar_file,  # Adding TAR file path to data
#                             "Query Path": html_member.name 
#                         })
#                     else:
#                         print(f"⚠️ No screenshot found for URL {url}.")
#                 else:
#                     print(f"⚠️ URL extraction failed for {html_path}")
#             else:
#                 print(f"⚠️ No HTML file found in TAR file {tar_file}.")
#     except Exception as e:
#         print(f"❌ Error processing TAR file: {e}")

#     print(f"🔢 Total screenshots extracted so far: {len(matched_data)}/{MAX_SCREENSHOTS}")

# # Step 3: Save matched data to CSV
# df = pd.DataFrame(matched_data)
# df.to_csv(OUTPUT_CSV, index=False)
# print(f"\n✅ Matched screenshot URLs saved to {OUTPUT_CSV}")


# import os
# import tarfile
# import random
# import pandas as pd
# from bs4 import BeautifulSoup

# BASE_DIRECTORY = "/mnt/ceph/storage/data-in-progress/data-research/web-search/affiliate-serp-crawls"
# SCREENSHOT_OUTPUT_DIR = "screenshots"  # Directory to store extracted screenshots
# OUTPUT_CSV = "screenshot_urls.csv"
# MAX_SCREENSHOTS = 100  # The number of unique screenshots to extract

# def extract_url_from_html(html_path):
#     """Extracts URL from HTML files."""
#     try:
#         with open(html_path, 'r', encoding='utf-8') as f:
#             soup = BeautifulSoup(f, 'html.parser')
#             # Try to find the canonical URL or OG URL
#             url = soup.find('meta', {'property': 'og:url'})
#             if url:
#                 return url.get('content')
#             url = soup.find('link', {'rel': 'canonical'})
#             if url:
#                 return url.get('href')
#             # If no meta or canonical tag, return the base URL from the HTML
#             return soup.find('base', {'href': True}).get('href', '#')
#     except Exception as e:
#         print(f"Error extracting URL from {html_path}: {e}")
#         return None

# # Scan directories for TAR files
# tar_files = []
# for root, dirs, files in os.walk(BASE_DIRECTORY):
#     for file in files:
#         if file.endswith(".tar"):
#             tar_files.append(os.path.join(root, file))

# print(f"✅ Found {len(tar_files)} TAR files.")

# # Step 2: Extract screenshots and URLs from TAR files
# matched_data = []
# while len(matched_data) < MAX_SCREENSHOTS and tar_files:
#     tar_file = random.choice(tar_files)  # Pick a random TAR file
#     tar_files.remove(tar_file)  # Remove from available TAR files

#     print(f"\n🔄 Processing TAR File: {tar_file}")

#     try:
#         with tarfile.open(tar_file, "r") as tar:
#             # Searching for any .html file in the hit-* folders (not inside snapshot)
#             html_files = [m for m in tar.getmembers() if m.name.endswith(".html") and 'hit-' in m.name]
#             if html_files:
#                 html_member = html_files[0]  # Pick the first HTML file found
#                 html_path = os.path.join(SCREENSHOT_OUTPUT_DIR, f"html_{len(matched_data)}.html")
#                 os.makedirs(SCREENSHOT_OUTPUT_DIR, exist_ok=True)
#                 with tar.extractfile(html_member) as src, open(html_path, "wb") as dst:
#                     dst.write(src.read())
                
#                 print(f"✅ Extracted HTML: {html_path}")
                
#                 # Extract URL from the HTML file
#                 url = extract_url_from_html(html_path)
#                 if url:
#                     print(f"✅ Extracted URL: {url}")
                    
#                     # Searching for screenshot in the snapshot folder
#                     snapshot_files = [m for m in tar.getmembers() if m.name.endswith("screenshot.png") and 'snapshot' in m.name]
#                     if snapshot_files:
#                         screenshot_member = snapshot_files[0]  # Pick the first screenshot found
#                         screenshot_path = os.path.join(SCREENSHOT_OUTPUT_DIR, f"screenshot_{len(matched_data)}.png")
#                         with tar.extractfile(screenshot_member) as src, open(screenshot_path, "wb") as dst:
#                             dst.write(src.read())
#                         print(f"✅ Extracted Screenshot: {screenshot_path}")
#                         matched_data.append({
#                             "Screenshot": screenshot_path,
#                             "URL": url
#                         })
#                     else:
#                         print(f"⚠️ No screenshot found for URL {url}.")
#                 else:
#                     print(f"⚠️ URL extraction failed for {html_path}")
#             else:
#                 print(f"⚠️ No HTML file found in TAR file {tar_file}.")
#     except Exception as e:
#         print(f"❌ Error processing TAR file: {e}")

#     print(f"🔢 Total screenshots extracted so far: {len(matched_data)}/{MAX_SCREENSHOTS}")

# # Step 3: Save matched data to CSV
# df = pd.DataFrame(matched_data)
# df.to_csv(OUTPUT_CSV, index=False)
# print(f"\n✅ Matched screenshot URLs saved to {OUTPUT_CSV}")




# import os
# import tarfile
# import random
# import re
# import pandas as pd
# from fastwarc import ArchiveIterator

# # ✅ Base directory where TAR (screenshots) and WARC (URLs) are stored
# BASE_DIRECTORY = "/mnt/ceph/storage/data-in-progress/data-research/web-search/affiliate-serp-crawls"
# SCREENSHOT_OUTPUT_DIR = "screenshots"  # Directory to store extracted screenshots
# OUTPUT_CSV = "screenshot_urls.csv"
# MAX_SCREENSHOTS = 100  # The number of unique screenshots to extract

# # ✅ Step 1: Scan Base Directory for TAR (Screenshots) and WARC (URLs)
# print("\n🔍 Scanning base directory for TAR and WARC files...")

# tar_files = []
# warc_files = []

# for root, dirs, files in os.walk(BASE_DIRECTORY):
#     for file in files:
#         file_path = os.path.join(root, file)
#         if file.endswith(".tar"):
#             tar_files.append(file_path)
#         elif file.endswith(".warc.gz"):
#             warc_files.append(file_path)

# print(f"✅ Found {len(tar_files)} TAR files (Screenshots).")
# print(f"✅ Found {len(warc_files)} WARC files (URLs).")

# # ✅ Step 2: Extract Query IDs from filenames
# def extract_query_id(filepath):
#     """Extracts query ID from TAR or WARC filename."""
#     match = re.search(r"query-(\d+)", filepath)
#     return match.group(1) if match else None

# tar_queries = {extract_query_id(t): t for t in tar_files if extract_query_id(t)}
# warc_queries = {extract_query_id(w): w for w in warc_files if extract_query_id(w)}

# # ✅ Step 3: Find all queries that have both screenshots & URLs
# common_queries = list(set(tar_queries.keys()) & set(warc_queries.keys()))

# print(f"\n✅ Found {len(common_queries)} queries with both screenshots & URLs.")

# # ✅ Step 4: Extract Screenshots & URLs from Matching Queries
# matched_data = []
# processed_queries = set()  # Keep track of queries that have been processed

# while len(matched_data) < MAX_SCREENSHOTS and common_queries:
#     query = random.choice(common_queries)  # Pick a random query
#     common_queries.remove(query)  # Remove from available queries
#     processed_queries.add(query)

#     tar_file = tar_queries.get(query)
#     warc_file = warc_queries.get(query)

#     if not tar_file or not warc_file:
#         print(f"⚠️ Skipping Query {query}: Missing TAR or WARC file.")
#         continue  # Skip queries without both files

#     print(f"\n🔄 Processing Query {query}")
#     print(f"📂 TAR File: {tar_file}")
#     print(f"🌍 WARC File: {warc_file}")

#     # ✅ Extract one random screenshot from TAR file
#     try:
#         with tarfile.open(tar_file, "r") as tar:
#             png_files = [m for m in tar.getmembers() if m.name.endswith(".png")]
#             if png_files:
#                 screenshot_member = random.choice(png_files)  # Pick a random screenshot
#                 extracted_screenshot_path = os.path.join(SCREENSHOT_OUTPUT_DIR, f"screenshot_{len(matched_data)}.png")

#                 os.makedirs(SCREENSHOT_OUTPUT_DIR, exist_ok=True)
#                 with tar.extractfile(screenshot_member) as src, open(extracted_screenshot_path, "wb") as dst:
#                     dst.write(src.read())

#                 print(f"✅ Extracted Screenshot: {extracted_screenshot_path}")
#             else:
#                 print(f"⚠️ No PNG found in TAR file for Query {query}. Skipping.")
#                 continue  # Skip to the next query

#     except Exception as e:
#         print(f"❌ Error extracting screenshot from TAR file: {e}")
#         continue

#     # ✅ Extract URL from WARC file
#     extracted_url = None
#     try:
#         with open(warc_file, "rb") as stream:
#             for record in ArchiveIterator(stream):
#                 if record.headers["WARC-Type"] == "response":
#                     url = record.headers.get("WARC-Target-URI", None)
#                     content_type = record.http_headers.get("Content-Type", "").lower()

#                     if content_type and "image/png" in content_type:
#                         extracted_url = url
#                         print(f"✅ Matched Screenshot URL: {extracted_url}")
#                         break  # Stop after finding the first valid URL

#     except Exception as e:
#         print(f"❌ Error processing WARC file: {e}")

#     if extracted_url:
#         matched_data.append({
#             "Query ID": query,
#             "WARC File": warc_file,
#             "Original URL": extracted_url,
#             "Extracted Screenshot": extracted_screenshot_path
#         })

#     print(f"🔢 Total screenshots extracted so far: {len(matched_data)}/{MAX_SCREENSHOTS}")

# # ✅ Step 5: Save Matched Data to CSV
# df = pd.DataFrame(matched_data)
# df.to_csv(OUTPUT_CSV, index=False)
# print(f"\n✅ Matched screenshot URLs saved to {OUTPUT_CSV}")
