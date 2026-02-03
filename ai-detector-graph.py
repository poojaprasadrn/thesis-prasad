# import pandas as pd
# #from elasticsearch import Elasticsearch, helpers
# import csv
# import os

# # # === CONFIG ===
# # CLASSIFIED_RESULTS_CSV = "/mnt/ceph/storage/data-tmp/current/yili5634/classified_results_2.csv"
# # ES_HOST = "https://elasticsearch.srv.webis.de/"
# # ES_USERNAME = "yili5634"
# # ES_PASSWORD = "KqAk50zmz80BSjtn"
# # ES_INDEX = "wstud_yili5634_ai_classification_results"


# # import pandas as pd
# # import csv

# # # File paths
# # cleaned_file = "/mnt/ceph/storage/data-tmp/current/yili5634/classified_cleaned.csv"
# # master_file = "/mnt/ceph/storage/data-tmp/current/yili5634/master_dataset.csv"
# # output_file = "/mnt/ceph/storage/data-tmp/current/yili5634/classified_cleaned_with_timestamp.csv"

# # # STEP 1: Load master dataset once
# # print("📥 Loading master dataset...")
# # df_master = pd.read_csv(master_file, usecols=["URL", "Timestamp"])

# # # Prepare output
# # header_written = False
# # processed_count = 0
# # chunk_size = 100000
# # buffer = []

# # print("🔄 Processing cleaned file in chunks...")

# # with open(cleaned_file, newline='', encoding='utf-8') as infile, \
# #      open(output_file, "w", newline='', encoding='utf-8') as outfile:

# #     reader = csv.reader(infile)
# #     writer = csv.writer(outfile)

# #     # Read header once
# #     header = next(reader)
# #     new_header = header + ["Timestamp"]
# #     writer.writerow(new_header)

# #     for row in reader:
# #         # Fix malformed rows
# #         if len(row) == 4:
# #             fixed_row = row
# #         elif len(row) > 4:
# #             merged_url = ','.join(row[:len(row)-3]).strip('"')
# #             fixed_row = [merged_url] + row[-3:]
# #         else:
# #             continue

# #         buffer.append(fixed_row)
# #         processed_count += 1

# #         # If enough for one chunk
# #         if len(buffer) >= chunk_size:
# #             df_chunk = pd.DataFrame(buffer, columns=header)
# #             df_chunk["URL"] = df_chunk["url"].str.strip('"')
# #             df_chunk.drop(columns=["url"], inplace=True)

# #             df_merged = df_chunk.merge(df_master, on="URL", how="left")
# #             df_merged.insert(0, "url", df_merged["URL"])  # restore original column
# #             df_merged.drop(columns=["URL"], inplace=True)

# #             writer.writerows(df_merged.values.tolist())
# #             print(f"✅ Processed {processed_count} rows...")

# #             buffer = []  # clear buffer

# #     # Final leftover buffer
# #     if buffer:
# #         df_chunk = pd.DataFrame(buffer, columns=header)
# #         df_chunk["URL"] = df_chunk["url"].str.strip('"')
# #         df_chunk.drop(columns=["url"], inplace=True)

# #         df_merged = df_chunk.merge(df_master, on="URL", how="left")
# #         df_merged.insert(0, "url", df_merged["URL"])
# #         df_merged.drop(columns=["URL"], inplace=True)

# #         writer.writerows(df_merged.values.tolist())
# #         print(f"✅ Final chunk processed. Total rows: {processed_count}")

# # print(f"🎉 All done! Merged file with timestamps saved at:\n{output_file}")


# import pandas as pd
# from elasticsearch import Elasticsearch, helpers

# # === CONFIG ===
# ES_HOST = "https://elasticsearch.srv.webis.de/"
# ES_USERNAME = "yili5634"
# ES_PASSWORD = "KqAk50zmz80BSjtn"
# ES_INDEX = "wstud_yili5634_ai_classification_results"

# INPUT_FILE = "/mnt/ceph/storage/data-tmp/current/yili5634/classified_results_v1.csv"


# # === 1. Load CSV ===
# df = pd.read_csv(INPUT_FILE, usecols=["url", "timestamp", "ai_label", "ai_probability"])
# df.dropna(subset=["timestamp"], inplace=True)
# print(f"Initial rows loaded: {len(df)}")
# print(df.head(3))

# # Properly handle 'Z' and convert to ISO
# df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors='coerce')
# df = df[df["timestamp"].notnull()]
# df["timestamp"] = df["timestamp"].dt.strftime('%Y-%m-%dT%H:%M:%S')

# print("Timestamps parsed successfully:")
# print(df["timestamp"].head(3))

# # === 2. Connect to Elasticsearch ===
# es = Elasticsearch(
#     hosts=[ES_HOST],
#     basic_auth=(ES_USERNAME, ES_PASSWORD),
#     verify_certs=True
# )
# if not es.indices.exists(index=ES_INDEX):
#     es.indices.create(index=ES_INDEX)
#     print(f"🆕 Index '{ES_INDEX}' created.")

# # === 3. Bulk Upload ===
# actions = [
#     {
#         "_index": ES_INDEX,
#         "_source": {
#             "url": row["url"],
#             "timestamp": row["timestamp"],
#             "ai_label": row["ai_label"],
#             "ai_probability": float(row["ai_probability"]),
#         }
#     }
#     for _, row in df.iterrows()
# ]

# print(f"📤 Uploading {len(actions)} entries...")
# helpers.bulk(es, actions)
# print(f"✅ Done uploading to index: {ES_INDEX}")





# import pandas as pd
# import re

# # Load master dataset
# df = pd.read_csv("/mnt/ceph/storage/data-tmp/current/yili5634/master_dataset.csv", usecols=["File Path", "URL"], on_bad_lines='skip')

# # Extract date, engine, and crawl type (e.g., gpc, organic, etc.)
# def extract_info(path):
#     path = str(path)
#     date_match = re.search(r"affiliate-serp-crawls/(\d{4}-\d{2}-\d{2})", path)
#     engine_match = re.search(r"affiliate-serp-crawls/\d{4}-\d{2}-\d{2}-([a-zA-Z]+)", path)
#     crawl_type_match = re.search(r"affiliate-serp-crawls/.+?/(gpc|organic|news|shopping|.*?)/", path)
    
#     date = date_match.group(1) if date_match else None
#     engine = engine_match.group(1) if engine_match else None
#     crawl_type = crawl_type_match.group(1) if crawl_type_match else "unknown"
    
#     return pd.Series([date, engine, crawl_type])

# df[["Crawl Date", "Search Engine", "Crawl Type"]] = df["File Path"].apply(extract_info)
# df = df.dropna(subset=["Crawl Date", "Search Engine"])

# # === Aggregated counts ===
# # 1. URLs per (date, engine, crawl_type)
# summary = (
#     df.groupby(["Crawl Date", "Search Engine", "Crawl Type"])
#     .agg(
#         URL_Count=("URL", "count"),
#         Unique_URLs=("URL", "nunique")
#     )
#     .reset_index()
# )
# summary.to_csv("crawl_stats_by_date_engine_type.csv", index=False)

# # 2. Total URLs per search engine
# engine_totals = df["Search Engine"].value_counts().reset_index()
# engine_totals.columns = ["Search Engine", "Total URLs"]
# engine_totals.to_csv("total_urls_per_engine.csv", index=False)

# # 3. Unique URLs
# total_urls = len(df)
# unique_urls = df["URL"].nunique()

# # === Print overview ===
# print(f"📦 Total URLs: {total_urls}")
# print(f"🔍 Unique URLs: {unique_urls}")
# print("✅ Per-crawl stats saved to: crawl_stats_by_date_engine_type.csv")
# print("✅ Search engine totals saved to: total_urls_per_engine.csv")


# import pandas as pd
# import re

# # Load classified results CSV
# classified_path = "/mnt/ceph/storage/data-tmp/current/yili5634/classified_results_v1.csv"
# df = pd.read_csv(classified_path, on_bad_lines='skip')

# # Parse and clean
# df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
# df = df.dropna(subset=["timestamp", "path", "ai_label"])
# df["Crawl Date"] = df["timestamp"].dt.date.astype(str)

# # Extract search engine and crawl type from WARC path
# def extract_engine_and_type(path):
#     path = str(path)
#     engine_match = re.search(r"affiliate-serp-crawls/\d{4}-\d{2}-\d{2}-([a-zA-Z]+)", path)
#     crawl_type_match = re.search(r"affiliate-serp-crawls/.+?/(gpc|organic|news|shopping|.*?)\/", path)
    
#     engine = engine_match.group(1).lower() if engine_match else "unknown"
#     crawl_type = crawl_type_match.group(1).lower() if crawl_type_match else "unknown"
#     return pd.Series([engine, crawl_type])

# df[["Search Engine", "Crawl Type"]] = df["path"].apply(extract_engine_and_type)

# # Standardize AI labels (just in case)
# df["ai_label"] = df["ai_label"].str.strip().str.lower()

# # === Aggregation ===
# summary = (
#     df.groupby(["Crawl Date", "Search Engine", "Crawl Type"])
#     .agg(
#         Total_URLs=("url", "count"),
#         Unique_URLs=("url", "nunique"),
#         AI_Detected=("ai_label", lambda x: (x == "ai detected").sum()),
#         Human_Detected=("ai_label", lambda x: (x == "human").sum())
#     )
#     .reset_index()
# )

# # Save to CSV
# summary.to_csv("crawl_stats_by_date_engine_type_with_ai.csv", index=False)

# # Print basic overview
# print(f"📦 Total Classified URLs: {len(df)}")
# print(f"🔍 Unique URLs: {df['url'].nunique()}")
# print("✅ Per-crawl stats (with AI labels) saved to: crawl_stats_by_date_engine_type_with_ai.csv")


import time
import pandas as pd
import numpy as np
from joblib import load
from tqdm import tqdm
import gc

CSV_PATH = "/mnt/ceph/storage/data-tmp/current/yili5634/startpage_10_per_date_v1.csv"
URL_COL = "URL"
TEXT_COL = "Text"
N_SAMPLES = 5000
CHUNK_SIZE = 1000
MAX_DOC_CHARS = 15000
LOWERCASE = True
AGG_METHOD = "mean"

print("Loading model and vectorizer (read-only)...")
model = load("best_model.pkl", mmap_mode="r")
vectorizer = load("best_vectorizer.pkl", mmap_mode="r")
print("✅ Model and vectorizer loaded.\n")

# ✅ Instead of reading entire file, read small chunk
print("Reading a lightweight preview from CSV...")
df_iter = pd.read_csv(CSV_PATH, usecols=[URL_COL, TEXT_COL], chunksize=10000)
df = next(df_iter)  # only first 100 rows
df = df.dropna(subset=[TEXT_COL])
sample_df = df.sample(n=min(N_SAMPLES, len(df)), random_state=42).reset_index(drop=True)
print(f"✅ Loaded {len(sample_df)} samples for testing.\n")

def predict_in_chunks(text):
    if LOWERCASE:
        text = text.lower()
    text = text[:MAX_DOC_CHARS]
    words = text.split()
    if not words:
        return 0.0
    chunk_preds = []
    for i in range(0, len(words), CHUNK_SIZE):
        chunk = " ".join(words[i:i + CHUNK_SIZE])
        try:
            X = vectorizer.transform([chunk])
            pred_prob = model.decision_function(X)[0]
            chunk_preds.append(pred_prob)
        except Exception:
            continue
    if not chunk_preds:
        return 0.0
    return float(np.mean(chunk_preds)) if AGG_METHOD == "mean" else float(np.median(chunk_preds))

times, results = [], []

for i, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
    url = str(row[URL_COL])
    text = str(row[TEXT_COL]).strip()
    if not text or text.lower() in ["nan", "none"]:
        continue

    start = time.time()
    ai_score = predict_in_chunks(text)
    end = time.time()

    elapsed_ms = (end - start) * 1000
    times.append(elapsed_ms)
    results.append((url, round(ai_score, 3), round(elapsed_ms, 2)))

    del text
    gc.collect()

if results:
    print("\n================ Timing Summary ================\n")
    for url, score, t in results:
        print(f"{url[:70]}... → {t:.2f} ms (AI score = {score})")
    avg = np.mean(times)
    print(f"\n✅ Average inference time: {avg:.2f} ms per URL ({len(results)} samples)\n")
else:
    print("❌ No valid samples processed.")
