# import pandas as pd
# from urllib.parse import urlparse
# import matplotlib.pyplot as plt

# # === CONFIG ===
# RESULTS_CSV = "/mnt/ceph/storage/data-tmp/current/yili5634/classified_results_v1.csv"
# DOMAIN_AGG_CSV = "/mnt/ceph/storage/data-tmp/current/yili5634/domain_aggregated_ai_likeness.csv"

# # === Load results
# df = pd.read_csv(RESULTS_CSV, on_bad_lines="skip")
# df["ai_probability"] = pd.to_numeric(df["ai_probability"], errors="coerce")

# # === Domain extraction
# df["domain"] = df["url"].apply(lambda x: urlparse(str(x)).netloc)

# # === Timestamp and month extraction
# df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
# df = df.dropna(subset=["timestamp"])
# df["month"] = df["timestamp"].dt.to_period("M")

# # === Remove duplicate URLs (keep only one entry per URL)
# df = df.drop_duplicates(subset=["url"])

# # === Domain-level aggregation
# domain_stats = (
#     df.groupby("domain")
#     .agg(
#         unique_url_count=("url", "count"),
#         ai_count=("ai_label", lambda x: (x == "AI Detected").sum()),
#         human_count=("ai_label", lambda x: (x == "Human").sum()),
#         median_ai_probability=("ai_probability", "median")
#     )
#     .reset_index()
# )

# # Save domain-level CSV
# domain_stats.to_csv(DOMAIN_AGG_CSV, index=False)

# # Group by month and compute statistics
# monthly_grouped = df.groupby("month")["ai_probability"]
# monthly_stats = pd.DataFrame({
#     "mean": monthly_grouped.mean(),
#     "median": monthly_grouped.median(),
#     "p25": monthly_grouped.quantile(0.25),
#     "p75": monthly_grouped.quantile(0.75),
#     "p95": monthly_grouped.quantile(0.95),
# })

# # Plot
# plt.figure(figsize=(14, 6))
# monthly_stats["mean"].plot(label="Mean", marker='o', color="blue")
# monthly_stats["median"].plot(label="Median (50th)", marker='s', color="green")
# monthly_stats["p75"].plot(label="75th Percentile", linestyle="--", color="orange")
# monthly_stats["p95"].plot(label="95th Percentile", linestyle="--", color="red")

# plt.title("AI-likeness Score Distribution Over Time")
# plt.xlabel("Month")
# plt.ylabel("AI Probability")
# plt.xticks(rotation=45)
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.legend()
# plt.tight_layout()

# output_path = "plot_ai_percentile_mean_median_by_month.png"
# plt.savefig(output_path, dpi=300)
# plt.show()

# Re-import necessary libraries after code execution state reset
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the classified results CSV
# csv_path = "/mnt/ceph/storage/data-tmp/current/yili5634/classified_results_v1.csv"
# df = pd.read_csv(csv_path, on_bad_lines='skip')

# # Standardize search engine names
# def normalize_engine(engine):
#     engine = str(engine).lower()
#     if 'startpage' in engine or 'gpc' in engine:
#         return 'google'
#     elif 'ddg' in engine:
#         return 'ddg'
#     elif 'bing' in engine:
#         return 'bing'
#     else:
#         return 'other'

# df["search_engine"] = df["search_engine"].apply(normalize_engine)

# # Drop invalid timestamps
# df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
# df = df.dropna(subset=["timestamp"])
# df["month"] = df["timestamp"].dt.to_period("M").astype(str)

# # Filter only valid engines
# df = df[df["search_engine"].isin(["google", "ddg", "bing"])]

# # Ensure AI probability is numeric
# df["ai_probability"] = pd.to_numeric(df["ai_probability"], errors="coerce")

# # === 1️⃣ Line Plot: Mean and Median AI Probability per Month per Engine
# agg_stats = df.groupby(["month", "search_engine"])["ai_probability"].agg(["mean", "median"]).reset_index()

# # === 2️⃣ Bar Plot: AI-labeled % per Engine per Month
# label_counts = df.groupby(["month", "search_engine", "ai_label"]).size().unstack(fill_value=0).reset_index()
# label_counts["total"] = label_counts.get("AI Detected", 0) + label_counts.get("Human", 0)
# label_counts["ai_percent"] = 100 * label_counts.get("AI Detected", 0) / label_counts["total"]


# # Plot 1: Line plot of mean and median
# plt.figure(figsize=(14, 6))
# sns.lineplot(data=agg_stats, x="month", y="mean", hue="search_engine", marker="o", linewidth=2, label="Mean")
# sns.lineplot(data=agg_stats, x="month", y="median", hue="search_engine", marker="s", linewidth=2, linestyle="--", legend=False)
# plt.title("Mean and Median AI Probability per Search Engine Over Time")
# plt.xlabel("Month")
# plt.ylabel("AI Probability")
# plt.xticks(rotation=45)
# plt.grid(True, linestyle="--", alpha=0.6)
# plt.ylim(0, 0.2)
# plt.tight_layout()
# plt.savefig("plot_mean_median_ai_prob_per_engine.png", dpi=300)
# plt.show()

# # Plot 2: Bar plot of AI percentage
# plt.figure(figsize=(14, 6))
# sns.barplot(data=label_counts, x="month", y="ai_percent", hue="search_engine")
# plt.title("Percentage of AI-Labeled Results per Search Engine per Month")
# plt.xlabel("Month")
# plt.ylabel("AI-Labeled Percentage (%)")
# plt.xticks(rotation=45)
# plt.grid(True, linestyle="--", alpha=0.5)
# plt.tight_layout()
# plt.savefig("plot_ai_label_percentage_bar_per_engine.png", dpi=300)
# plt.show()


# === Monthly AI trend plot (Median instead of Mean)
# monthly_median = df.groupby("month")["ai_probability"].median()

# plt.figure(figsize=(10, 5))
# monthly_median.plot(marker='o', linestyle='-', color='teal')
# plt.title("Median AI-likeness Over Time")
# plt.xlabel("Month")
# plt.ylabel("Median AI Probability")
# plt.xticks(rotation=45)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.tight_layout()
# plt.savefig("plot_ai_median_by_month.png", dpi=300)
# plt.show()

# percentiles = df.groupby("month")["ai_probability"].quantile([0.9, 0.95]).unstack()

# # Plot
# plt.figure(figsize=(12, 6))
# plt.plot(percentiles.index.astype(str), percentiles[0.9], marker='o', label="90th Percentile", color='orange')
# plt.plot(percentiles.index.astype(str), percentiles[0.95], marker='s', label="95th Percentile", color='red')
# plt.title("AI Probability 90th and 95th Percentile Over Time")
# plt.xlabel("Month")
# plt.ylabel("AI Probability")
# plt.xticks(rotation=45)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend()
# plt.tight_layout()
# plt.savefig("plot_ai_percentiles_by_month.png", dpi=300)


# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import os
# from matplotlib.ticker import LogLocator

# # === Config paths
# RESULTS_CSV = "/mnt/ceph/storage/data-tmp/current/yili5634/classified_results_v1.csv"
# DOMAIN_AGG_CSV = "/mnt/ceph/storage/data-tmp/current/yili5634/domain_aggregated_ai_likeness.csv"

# # === Create output folder
# output_dir = "/home/yili5634/Desktop/thesis-pooja"
# os.makedirs(output_dir, exist_ok=True)

# # === Load main classification results
# df = pd.read_csv(RESULTS_CSV, quotechar='"', on_bad_lines='skip')
# df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
# df = df.dropna(subset=["timestamp"])
# df["month"] = df["timestamp"].dt.to_period("M")

# # === Monthly AI trend plot
# monthly_avg = df.groupby("month")["ai_probability"].mean()
# monthly_avg.plot(title="AI-likeness Over Time", ylabel="Avg AI Probability", xlabel="Month", rot=45)
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "plot_monthly_avg_ai_prob.png"), dpi=300)
# plt.show()


# 1. Histogram with log y-axis
# plt.figure(figsize=(8, 5))
# plt.hist(df_domains["mean"], bins=50, color='skyblue', edgecolor='black')
# plt.xlabel("Average AI Probability")
# plt.ylabel("Number of Domains (log scale)")
# plt.yscale("log")
# plt.title("Log-Scale Distribution of Domain-Level AI-likeness Scores")
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "plot_domain_score_distribution_log.png"), dpi=300)
# plt.close()

# # 2. Top AI domains (reduce to 10 for better visual)
# top_df = df_domains[df_domains["count"] >= 10].sort_values("mean", ascending=False).head(10)
# plt.figure(figsize=(10, 6))
# sns.barplot(y="domain", x="mean", data=top_df, palette="Oranges_r")
# plt.xlabel("Mean AI Probability")
# plt.title("Top 10 High-Confidence AI Domains (≥10 pages)")
# plt.xlim(0.9, 1.0)
# for index, value in enumerate(top_df["mean"]):
#     plt.text(value - 0.005, index, f"{value:.3f}", va='center', ha="right", color="black")
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "plot_top_ai_domains_10.png"), dpi=300)
# plt.close()

# # 3. CDF with vertical lines
# plt.figure(figsize=(8, 5))
# sns.ecdfplot(df_domains["mean"], color='purple', linewidth=2)
# plt.xlabel("Average AI Probability")
# plt.ylabel("Cumulative Distribution")
# plt.title("CDF of Domain-Level AI-likeness Scores")
# plt.axvline(x=0.5, color='red', linestyle='--', label='AI Threshold (0.5)')
# plt.axvline(x=0.9, color='green', linestyle='--', label='High Confidence (0.9)')
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "plot_domain_score_distribution_cdf_thresh.png"), dpi=300)
# plt.close()

# # 4. Violin plot of AI probability over time
# df = pd.read_csv(RESULTS_CSV, quotechar='"', on_bad_lines='skip')
# df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
# df = df.dropna(subset=["timestamp"])
# df["quarter"] = df["timestamp"].dt.to_period("Q").astype(str)
# quarterly_summary = df.groupby("quarter")["ai_probability"].agg(["mean", "median"]).reset_index()

# plt.figure(figsize=(14, 5))
# sns.lineplot(data=quarterly_summary, x="quarter", y="median", marker="o", label="Median", color="purple")
# sns.lineplot(data=quarterly_summary, x="quarter", y="mean", marker="s", label="Mean", color="green")
# plt.xticks(rotation=45)
# plt.ylim(0, 1)
# plt.xlabel("Quarter")
# plt.ylabel("AI Probability")
# plt.title("Mean & Median AI Probability Over Time", fontsize=14, weight='bold')
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "plot_ai_mean_median_by_quarter.png"), dpi=300)
# plt.close()


# import pandas as pd
# from elasticsearch import Elasticsearch, helpers
# import hashlib

# # === CONFIG ===
# ES_HOST = "https://elasticsearch.srv.webis.de/"
# ES_USERNAME = "yili5634"
# ES_PASSWORD = "KqAk50zmz80BSjtn"

# # === CSVs and Index Names ===
# FILES_AND_INDICES = [
#     {
#         "csv": "/mnt/ceph/storage/data-tmp/current/yili5634/classified_results_v1.csv",
#         "index": "wstud_yili5634_ai_classification_results"
#     },
#     {
#         "csv": "/mnt/ceph/storage/data-tmp/current/yili5634/classified_results_v1_enhanced.csv",
#         "index": "wstud_yili5634_ai_classification_enhanced"
#     },
#     {
#         "csv": "/mnt/ceph/storage/data-tmp/current/yili5634/domain_aggregated_ai_likeness.csv",
#         "index": "wstud_yili5634_ai_domain_stats"
#     }
# ]

# DELETE_OLD_INDEX = False  # ✅ Set True if you want to wipe and refresh index

# # === 1. Connect ===
# es = Elasticsearch(
#     hosts=[ES_HOST],
#     basic_auth=(ES_USERNAME, ES_PASSWORD),
#     verify_certs=True
# )

# # === 2. Function to compute unique ID ===
# def make_id(row):
#     joined = "".join(str(v) for v in row.values)
#     return hashlib.md5(joined.encode("utf-8")).hexdigest()

# # === 3. Upload function ===
# def upload_csv_to_index(csv_path, index_name):
#     print(f"\n📦 Uploading: {csv_path} ➜ Index: {index_name}")
    
#     df = pd.read_csv(csv_path, quotechar='"', on_bad_lines="skip")

#     df.dropna(how="all", inplace=True)  # Remove fully empty rows
#     df.fillna("", inplace=True)

#     # Special timestamp formatting
#     if "timestamp" in df.columns:
#         df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
#         df = df[df["timestamp"].notnull()]
#         df["timestamp"] = df["timestamp"].dt.strftime('%Y-%m-%dT%H:%M:%S')

#     if DELETE_OLD_INDEX and es.indices.exists(index=index_name):
#         es.indices.delete(index=index_name)
#         print(f"🗑️ Deleted index: {index_name}")
    
#     if not es.indices.exists(index=index_name):
#         es.indices.create(index=index_name)
#         print(f"🆕 Created index: {index_name}")

#     actions = []
#     for _, row in df.iterrows():
#         doc = row.to_dict()
#         doc_id = make_id(row)
#         actions.append({
#             "_index": index_name,
#             "_id": doc_id,
#             "_source": doc
#         })

#     helpers.bulk(es, actions, chunk_size=1000, max_retries=3, request_timeout=60)

#     print(f"✅ Uploaded {len(actions)} records to {index_name}")

# # === 4. Loop and Upload ===
# for item in FILES_AND_INDICES:
#     upload_csv_to_index(item["csv"], item["index"])

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from urllib.parse import urlparse
# import os
# import re
# from matplotlib.ticker import PercentFormatter

# output_dir = "/home/yili5634/Desktop/thesis-pooja"

# # Load the data
# df = pd.read_csv(
#     "/mnt/ceph/storage/data-tmp/current/yili5634/classified_results_v1.csv",
#     on_bad_lines="skip"
# )

# # Convert probability to float
# df["ai_probability"] = pd.to_numeric(df["ai_probability"], errors="coerce")

# # Convert timestamp and extract month
# df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
# df["month"] = df["timestamp"].dt.to_period("M").astype(str)

# # ✅ Extract search engine properly from path
# def extract_engine(path):
#     try:
#         match = re.search(r"/\d{4}-\d{2}-\d{2}-([a-zA-Z]+)/", path)
#         if match:
#             return match.group(1).lower()
#         return 
#     except Exception:
#         return "unknown"

# df["search_engine"] = df["path"].apply(extract_engine)

# # ✅ Group by month and search engine, compute mean
# mean_df = df.groupby(["month", "search_engine"])["ai_probability"].mean().reset_index()

# # ✅ Plotting
# plt.figure(figsize=(12, 6))
# sns.lineplot(data=mean_df, x="month", y="ai_probability", hue="search_engine", marker="o")
# plt.title("Mean AI Probability Over Time by Search Engine")
# plt.xlabel("Month")
# plt.ylabel("Mean AI Probability (0 = Human, 1 = AI)")
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "plot_ai_detection_mean_per_engine.png"))
# plt.show()

# #----------------------------------------------

# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# # Load dataset
# input_file = "/mnt/ceph/storage/data-tmp/current/yili5634/classified_results_v1.csv"
# df = pd.read_csv(input_file, on_bad_lines="skip")

# # Convert timestamp and extract month
# df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
# df["month"] = df["timestamp"].dt.to_period("M").astype(str)

# # Extract search engine from path
# def extract_engine(path):
#     engines = ["ddg", "bing", "startpage"]
#     for engine in engines:
#         if engine in str(path).lower():
#             return engine
#     return "unknown"

# df["search_engine"] = df["path"].apply(extract_engine)

# # Clean and normalize labels
# df["ai_label"] = df["ai_label"].str.strip().str.lower()

# # Deduplicate based on URL only
# df_unique = df.drop_duplicates(subset=["url"])

# # Group by month + engine + label
# grouped = df_unique.groupby(["search_engine", "month", "ai_label"]).size().unstack(fill_value=0).reset_index()
# grouped.columns.name = None

# # Add total and AAR (AI Attribution Rate)
# grouped["total"] = grouped.get("ai detected", 0) + grouped.get("human", 0)
# grouped["AAR"] = grouped.get("ai detected", 0) / grouped["total"]

# # Plot directory
# output_dir = "./ai_human_plots"
# os.makedirs(output_dir, exist_ok=True)

# # Plot per search engine
# for engine in grouped["search_engine"].unique():
#     engine_df = grouped[grouped["search_engine"] == engine].sort_values("month")
#     fig, ax1 = plt.subplots(figsize=(12, 6))

#     months = engine_df["month"]

#     # === Left axis (Human count)
#     ax1.bar(months, engine_df.get("human", 0), width=0.4, label="Human", color="skyblue", align="edge")
#     ax1.set_ylabel("Human URL Count", color="#0d417a")
#     ax1.tick_params(axis='y', labelcolor="#0d417a")
#     ax1.set_xlabel("Month")
#     ax1.tick_params(axis='x', rotation=45)

#     # === Second left axis (AI count - offset manually)
#     ax3 = ax1.twinx()  # Start with a twin
#     ax3.spines["left"].set_position(("axes", -0.10))  # Shift to left of ax1
#     ax3.spines["left"].set_visible(True)
#     ax3.yaxis.set_label_position("left")
#     ax3.yaxis.set_ticks_position("left")
#     ax3.bar(months, engine_df.get("ai detected", 0), width=-0.4, label="AI", color="salmon", align="edge")
#     ax3.set_ylabel("AI URL Count", color="salmon")
#     ax3.tick_params(axis='y', labelcolor="salmon")

#     # === Right axis (AAR line)
#     ax2 = ax1.twinx()
#     # Convert AAR to percentage
#     ax2.plot(months, engine_df["AAR"] * 100, color="black", marker="o", label="AAR")
#     ax2.set_ylabel("AI Attribution Rate (AAR) [%]", color="black")
#     ax2.tick_params(axis='y', labelcolor="black")
#     ax2.set_ylim(0, 10)  # 0% to 10%
#     ax2.yaxis.set_major_formatter(PercentFormatter())

#     # === Title and layout
#     plt.title(f"AI vs Human Unique URL Counts and AAR - {engine.capitalize()}")
#     fig.tight_layout()
#     fig.subplots_adjust(left=0.15)  # Adjust for extra y-axis

#     # === Save
#     filename = os.path.join(output_dir, f"ai_vs_human_threey_{engine}.png")
#     plt.savefig(filename)
#     plt.close()

# print(f"✅ Saved all updated three-axis plots in: {output_dir}")



# #-------------------------------------------------------------------


# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# # Load the data
# file_path = "crawl_stats_by_date_engine_type_with_ai.csv"
# df = pd.read_csv(file_path)

# # Format date and calculate duplicate URLs
# df["Crawl Date"] = pd.to_datetime(df["Crawl Date"])
# df["Month"] = df["Crawl Date"].dt.to_period("M").astype(str)
# df["Duplicate_URLs"] = df["Total_URLs"] - df["Unique_URLs"]

# # Create output folder
# os.makedirs("ai_human_plots", exist_ok=True)

# # Plot for each search engine
# for engine in df["Search Engine"].unique():
#     sub_df = df[df["Search Engine"] == engine]
#     monthly = sub_df.groupby("Month")[["Unique_URLs", "Duplicate_URLs"]].sum().reset_index()

#     # X axis positions
#     x = np.arange(len(monthly))
#     width = 0.35

#     fig, ax1 = plt.subplots(figsize=(10, 5))

#     # Left Y-axis: Unique URLs
#     bar1 = ax1.bar(x - width/2, monthly["Unique_URLs"], width, color="#A6CDD4", label="Unique URLs")
#     ax1.set_ylabel("Unique URLs", color="#4F7F87")
#     ax1.tick_params(axis='y', labelcolor="#4F7F87")

#     # Right Y-axis: Duplicate URLs
#     ax2 = ax1.twinx()
#     bar2 = ax2.bar(x + width/2, monthly["Duplicate_URLs"], width, color="#f3aab2", label="Duplicate URLs")
#     ax2.set_ylabel("Duplicate URLs", color="#93565C")
#     ax2.tick_params(axis='y', labelcolor="#93565C")

#     # X ticks and labels
#     ax1.set_xticks(x)
#     ax1.set_xticklabels(monthly["Month"], rotation=45)
#     ax1.set_xlabel("Month")

#     # Title and legends
#     plt.title(f"{engine.capitalize()} - Unique vs Duplicate URLs (Dual Axis)")
#     fig.tight_layout()

#     # Save
#     filename = f"ai_human_plots/dual_axis_bars_{engine.lower().replace(' ', '_')}.png"
#     plt.savefig(filename, dpi=300, bbox_inches="tight")
#     plt.close()

#----------------------------------------------------------------------


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# ======= CONFIG =======
CSV_PATH  = "/mnt/ceph/storage/data-tmp/current/yili5634/classified_results_v1.csv"
OUT_DIR   = "./ai_human_plots"   # plots will be saved here
ENGINES   = ["bing", "duckduckgo", "startpage"]   # engines to plot
# ======================

os.makedirs(OUT_DIR, exist_ok=True)

ENGINE_TOKENS = ["startpage","bing","duckduckgo","ddg"]

def robust_read_csv(p):
    return pd.read_csv(p, on_bad_lines="skip")

def extract_engine(path: str) -> str:
    p = str(path).lower()
    for tok in ENGINE_TOKENS:
        if tok in p:
            return "duckduckgo" if tok == "ddg" else tok
    m = re.search(r"/20\d{2}-\d{2}-\d{2}-([a-z0-9\-]+)/", p)
    if m:
        cand = m.group(1)
        if cand == "ddg":
            return "duckduckgo"
        return cand
    return "unknown"

def extract_crawl_date(path: str, fallback_ts: str | None) -> pd.Timestamp | pd.NaT:
    m = re.search(r"(20\d{2}-\d{2}-\d{2})", str(path))
    if m:
        return pd.to_datetime(m.group(1), errors="coerce")
    if fallback_ts:
        return pd.to_datetime(str(fallback_ts)[:10], errors="coerce")
    return pd.NaT

def normalize_label(v: str | None) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip().lower()
    if s in {"ai","ai detected","machine","synthetic","ml","1","true","yes"}:
        return "ai"
    if s in {"human","0","false","no"}:
        return "human"
    return ""

# ---------- Load & prepare ----------
df = robust_read_csv(CSV_PATH)

for col in ("url","path","timestamp","ai_label"):
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

df["timestamp"]  = pd.to_datetime(df["timestamp"], errors="coerce")
df["engine"]     = df["path"].apply(extract_engine)
df["crawl_date"] = df.apply(lambda r: extract_crawl_date(r.get("path",""), r.get("timestamp","")), axis=1)
df["month"]      = df["crawl_date"].values.astype("datetime64[M]")
df["label"]      = df["ai_label"].apply(normalize_label)

# Keep only engines of interest
df = df[df["engine"].isin(ENGINES)].copy()

# Decide label per (engine, month, url): any AI→ai; else any human→human; else drop
per_month_url = (
    df.groupby(["engine","month","url"], as_index=False)
      .agg(label=("label", lambda s: "ai" if (s=="ai").any()
                             else ("human" if (s=="human").any() else "")))
)
per_month_url = per_month_url[per_month_url["label"] != ""]

# Aggregate per engine-month
agg = (
    per_month_url.groupby(["engine","month"], as_index=False)
                 .agg(unique_urls=("url","nunique"),
                      ai_urls=("label", lambda s: int((s=="ai").sum())))
)
agg["total"] = agg["unique_urls"]
agg["AAR"]   = np.where(agg["total"] > 0, agg["ai_urls"] / agg["total"], np.nan)

# ---------- Separate plots per engine ----------
months_all = np.sort(agg["month"].unique())
x_pos = np.arange(len(months_all), dtype=float)

for eng in ENGINES:
    sub = agg[agg["engine"] == eng].set_index("month").reindex(months_all)

    # Series aligned to full month index
    unique_vals = sub["unique_urls"].fillna(0).values
    aar_vals    = sub["AAR"].values

    fig, ax_left = plt.subplots(figsize=(14, 6))

    # Bars: unique URLs (left y-axis)
    ax_left.bar(x_pos, unique_vals, width=0.8)
    ax_left.set_xlabel("Month")
    ax_left.set_ylabel("Unique URLs (per month)")
    ax_left.set_xticks(x_pos)
    ax_left.set_xticklabels([pd.Period(m, freq="M").strftime("%Y-%m") for m in months_all], rotation=45)

    # Line: AAR (right y-axis)
    ax_right = ax_left.twinx()
    ax_right.plot(x_pos, aar_vals, marker="o", linewidth=1.8)
    ax_right.set_ylabel("AI Attribution Rate")
    ax_right.set_ylim(0, 1)
    ax_right.yaxis.set_major_formatter(PercentFormatter(1.0))

    plt.title(f"Monthly Unique URLs (bars) and AAR (line) — {eng.capitalize()}")
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"monthly_unique_and_aar_{eng}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"✅ Saved: {out_path}")
