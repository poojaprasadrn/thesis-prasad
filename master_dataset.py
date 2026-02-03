import os
import csv
from warcio.archiveiterator import ArchiveIterator
from resiliparse.parse.html import HTMLTree
from resiliparse.extract.html2text import extract_plain_text

DATA_ROOT = '/mnt/ceph/storage/data-in-progress/data-research/web-search/affiliate-serp-crawls'
OUTPUT_CSV = '/mnt/ceph/storage/data-tmp/current/yili5634/master_dataset.csv'

def find_files(root, exts=('.warc', '.warc.gz')):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(exts):
                files.append(os.path.join(dirpath, f))
    return files

def extract_main_content_resiliparse(html_bytes):
    try:
        html_str = html_bytes.decode('utf-8', errors='ignore')
        tree = HTMLTree.parse(html_str)
        text = extract_plain_text(tree, preserve_formatting=False)
        return ' '.join(text.strip().split())
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ''

# === MAIN ===

warc_files = find_files(DATA_ROOT)

print(f"🔍 Found {len(warc_files)} WARC files")

fields = ['URL', 'Timestamp', 'File Path', 'Text']

with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f_out:
    writer = csv.DictWriter(f_out, fieldnames=fields)
    writer.writeheader()

    for f in warc_files:
        print(f"📁 Processing: {f}")
        try:
            with open(f, 'rb') as stream:
                for record in ArchiveIterator(stream):
                    if record.rec_type != 'response':
                        continue

                    url = record.rec_headers.get_header('WARC-Target-URI') or ''
                    warc_date = record.rec_headers.get_header('WARC-Date') or ''
                    content_type = record.http_headers.get_header('Content-Type') if record.http_headers else ''
                    #if record.http_headers and 'Last-Modified' in record.http_headers.headers:
                        #last_modified = record.http_headers.get_header('Last-Modified')
                    #else:
                        #last_modified = ''

                    # Only attempt HTML content
                    if 'html' not in (content_type or '').lower():
                        continue

                    html_content = record.content_stream().read()
                    text = extract_main_content_resiliparse(html_content)

                    if not text.strip():
                        continue  # skip empty

                    writer.writerow({
                        'URL': url,
                        'Timestamp': warc_date,
                        #'Last modified': last_modified,
                        'File Path': f,
                        'Text': text
                    })

                    print(f"✅ Extracted: {url} ({len(text.split())} words)")

        except Exception as e:
            print(f"⚠️ Error processing {f}: {e}")

print(f"\n🎉 DONE! Streamed all records safely to {OUTPUT_CSV}")
-------------------------------------------

!/usr/bin/env python3
-*- coding: utf-8 -*-

import os, re
import pandas as pd
from pathlib import Path

# ---------- CONFIG ----------
#CSV_PATH = "/mnt/ceph/storage/data-tmp/current/yili5634/candidates_balanced.csv"
CSV_PATH = "/mnt/ceph/storage/data-tmp/current/yili5634/startpage_10_per_date.csv"

ENGINE_TOKENS = [
    "startpage","bing", "ddg"
]
# ----------------------------

def extract_engine(path: str) -> str:
    p = str(path).lower()
    for tok in ENGINE_TOKENS:
        if tok in p:
            return tok
    return "unknown"

def extract_crawl_id(path: str) -> str:
    """
    Crawl ID = date-engine string from path, e.g. '2023-11-28-startpage'
    """
    m = re.search(r"(20\d{2}-\d{2}-\d{2}[^/\\]*)", str(path))
    return m.group(1) if m else "unknown"

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(CSV_PATH)

    df = pd.read_csv(CSV_PATH)

    if "URL" not in df.columns or "File Path" not in df.columns:
        raise ValueError("CSV must have 'URL' and 'File Path' columns")

    df["engine"] = df["File Path"].apply(extract_engine)
    df["crawl_id"] = df["File Path"].apply(extract_crawl_id)

    summary = (
        df.groupby(["engine","crawl_id"])
        .agg(total_urls=("URL","count"),
             unique_urls=("URL","nunique"))
        .reset_index()
        .sort_values(["engine","crawl_id"])
    )

    print("\n=== Per-crawl summary (total + unique URLs) ===")
    print(summary.to_string(index=False))

    # save as CSV too
    out_path = Path(CSV_PATH).with_suffix(".crawl_summary.csv")
    summary.to_csv(out_path, index=False)
    print(f"\n✅ Saved summary to {out_path}")

if __name__ == "__main__":
    main()

#----------------------------------------------

# import pandas as pd

# # Paths
# csv1 = "/mnt/ceph/storage/data-tmp/current/yili5634/classified_results_v1.csv"   # main file to append into
# csv2 = "/mnt/ceph/storage/data-tmp/current/yili5634/classified_results_v2.csv"   # file whose rows will be appended

# # Read both
# df1 = pd.read_csv(csv1)
# df2 = pd.read_csv(csv2)

# # Append
# df1 = pd.concat([df1, df2], ignore_index=True)

# # Overwrite csv1 with appended content
# df1.to_csv(csv1, index=False)

# print(f"✅ Appended {csv2} into {csv1}")


#!/usr/bin/env python3
# import sys
# import os

# def append_csv_inplace(csv_into: str, csv_from: str):
#     if not os.path.exists(csv_into):
#         raise FileNotFoundError(csv_into)
#     if not os.path.exists(csv_from):
#         raise FileNotFoundError(csv_from)

#     # Read headers (first line) of both files
#     with open(csv_into, "rb") as f1:
#         header1 = f1.readline()
#     with open(csv_from, "rb") as f2:
#         header2 = f2.readline()

#     if header1.strip() != header2.strip():
#         print("⚠️  Header mismatch between files.")
#         print("Into :", header1.decode(errors='ignore').strip())
#         print("From :", header2.decode(errors='ignore').strip())
#         print("Proceeding to append rows anyway (skipping file2 header).")

#     # Ensure csv_into ends with a newline so the first appended row isn’t glued
#     with open(csv_into, "rb+") as fout:
#         fout.seek(0, os.SEEK_END)
#         if fout.tell() > 0:
#             fout.seek(-1, os.SEEK_END)
#             last = fout.read(1)
#             if last != b"\n":
#                 fout.write(b"\n")

#     # Append everything from csv_from after its first line
#     with open(csv_into, "ab") as fout, open(csv_from, "rb") as fin:
#         fin.readline()  # skip header
#         # stream copy in chunks
#         while True:
#             chunk = fin.read(1024 * 1024)
#             if not chunk:
#                 break
#             fout.write(chunk)

#     print(f"✅ Appended '{csv_from}' into '{csv_into}'")

# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         print("Usage: python append_csv_inplace.py <into_csv> <from_csv>")
#         sys.exit(1)
#     append_csv_inplace(sys.argv[1], sys.argv[2])
