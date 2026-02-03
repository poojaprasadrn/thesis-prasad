import os
import re
import pandas as pd
from bs4 import BeautifulSoup
from warcio.archiveiterator import ArchiveIterator

# ✅ Paths
screenshots_dir = "rating/screenshots"  # Directory containing extracted screenshots
warc_base_dir = "/mnt/ceph/storage/data-in-progress/data-research/web-search/affiliate-serp-crawls"
output_file = "screenshot_urls_from_warc.csv"

# ✅ Find all screenshots
screenshot_files = [f for f in os.listdir(screenshots_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
if not screenshot_files:
    raise FileNotFoundError(f"❌ No screenshots found in {screenshots_dir}")

print(f"✅ Found {len(screenshot_files)} extracted screenshots.")

# ✅ Find all WARC files
warc_files = []
for root, _, files in os.walk(warc_base_dir):
    for file in files:
        if file.endswith(".warc.gz"):
            warc_files.append(os.path.join(root, file))

if not warc_files:
    raise FileNotFoundError(f"❌ No .warc.gz files found in {warc_base_dir} or its subdirectories.")

print(f"✅ Found {len(warc_files)} WARC files.")

# ✅ Step 2: Extract Image References from WARC
matched_urls = []

for warc_file in warc_files:
    print(f"🔍 Processing WARC file: {warc_file}")

    with open(warc_file, "rb") as f:
        for record in ArchiveIterator(f):
            if record.rec_type == "response":
                url = record.rec_headers.get_header("WARC-Target-URI")
                html_content = record.content_stream().read().decode("utf-8", errors="ignore")

                # ✅ Extract `<img src="...">` references
                soup = BeautifulSoup(html_content, "html.parser")
                img_tags = soup.find_all("img")

                for img in img_tags:
                    img_src = img.get("src")

                    if img_src:
                        # ✅ Get the image filename (strip paths)
                        img_filename = os.path.basename(img_src)

                        # ✅ Check if this image matches an extracted screenshot
                        for screenshot in screenshot_files:
                            if screenshot in img_filename:
                                matched_urls.append({
                                    "Screenshot": screenshot,
                                    "Image Reference": img_src,
                                    "URL": url
                                })
                                print(f"✅ Matched {screenshot} → {url}")

# ✅ Step 3: Save Matched URLs to CSV
df_urls = pd.DataFrame(matched_urls)
df_urls.to_csv(output_file, index=False)

print(f"✅ Screenshot URLs saved to {output_file}")
