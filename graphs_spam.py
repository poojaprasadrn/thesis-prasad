#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import pearsonr

# ========= CONFIG =========
SPAM_CSV = "/mnt/ceph/storage/data-tmp/current/yili5634/bert_predictions_binary_multiclass_v2.csv"
CANDIDATES_CSV = "/mnt/ceph/storage/data-tmp/current/yili5634/candidates_balanced.csv"
OUT_DIR = "./ai_human_plots_spam"
MONTH_LOOKBACK = 40

COLOR_SPAM = "#CC6677"
COLOR_GOOD = "#88CCEE"
COLOR_RATE = "#000000"
DPI = 600

IMPORTANT_DATES = [
    ("ChatGPT launch", "2022-11-30"),
    ("Bing Chat", "2023-02-07"),
    ("GPT-4", "2023-03-14"),
]

MIN_MONTH_TOTAL_BY_ENGINE = {"bing": 50, "duckduckgo": 50, "startpage": 50}
ENGINE_TOKENS = ["startpage", "bing", "duckduckgo", "ddg"]
os.makedirs(OUT_DIR, exist_ok=True)

# ========= HELPERS =========
def normalize_binary_class(x: str) -> str:
    if x is None:
        return ""
    x = str(x).strip().lower()
    return "review" if x == "review" else "other"

def normalize_review_type(x: str) -> str:
    if x is None:
        return ""
    x = str(x).strip().lower()
    if x in {"spam", "spam review"}:
        return "spam"
    if x in {"good", "good review"}:
        return "good"
    return "other"

def extract_engine_from_text(text: str) -> str:
    p = str(text).lower()
    m = re.search(r"/?20\d{2}-\d{2}-\d{2}-([a-z0-9\-]+)/", p)
    if m:
        cand = m.group(1)
        return "duckduckgo" if cand == "ddg" else cand
    for tok in ENGINE_TOKENS:
        if tok in p:
            return "duckduckgo" if tok == "ddg" else tok
    return "unknown"

def find_pathlike_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if any(h in c.lower() for h in ("path","file","dir","folder","source","location")):
            return c
    return None

# ========= PLOT =========
def add_vertical_markers(ax_right, months_all: pd.DatetimeIndex):
    if months_all.empty: return
    for label, d in IMPORTANT_DATES:
        ts = pd.to_datetime(d, errors="coerce")
        if pd.isna(ts): continue
        mstart = ts.to_period("M").to_timestamp()
        if not (months_all.min() <= mstart <= months_all.max()): continue
        idx = np.where(months_all == mstart)[0]
        if len(idx) == 0: continue
        x = int(idx[0])
        ax_right.axvline(x, color="#666666", linestyle="--", linewidth=1.1, alpha=0.9, zorder=2)
        y_top = ax_right.get_ylim()[1]
        ax_right.text(x - 0.2, y_top * 0.97, label,
                      rotation=90, va="top", ha="right",
                      fontsize=13, color="#444444",
                      backgroundcolor="white",
                      bbox=dict(facecolor="white", edgecolor="none", pad=0.3),
                      zorder=6)

def dual_axis_plot(months_all, good_vals, spam_vals, rate_vals, title, out_path, r_val=None, p_val=None):
    """
    Thesis-grade dual-axis plot:
      • Blue/Red bars: monthly counts of good vs spam reviews
      • Black line: observed spam rate (%)
      • Gray dashed line: linear trend (Pearson correlation)
      • r and p values annotated (✓ for significant, n.s. for non-significant)
      • Vertical lines: milestone events (ChatGPT, Bing Chat, GPT-4)
    """

    # === Layout ===
    fig, axL = plt.subplots(figsize=(16, 6.5), dpi=DPI)
    x = np.arange(len(months_all), dtype=float)

    # === Bars ===
    w = 0.4
    axL.bar(x - w/2, good_vals, width=w,
            color="#88CCEE", label="Good", alpha=0.95, zorder=1)
    axL.bar(x + w/2, spam_vals, width=w,
            color="#CC6677", label="Spam", alpha=0.95, zorder=1)

    # === X-axis labels ===
    labels = pd.Index(months_all).strftime("%Y-%m").tolist()
    step = max(1, int(np.ceil(len(labels) / 12)))  # ~12 ticks
    axL.set_xticks(x)
    axL.set_xticklabels([lab if i % step == 0 else "" for i, lab in enumerate(labels)],
                        rotation=45, fontsize=15)

    axL.set_ylabel("Unique Review URLs", fontsize=15, fontweight="bold")
    axL.set_xlabel("Month", fontsize=15, fontweight="bold")
    axL.tick_params(axis="y", labelsize=14)
    axL.grid(axis="y", alpha=0.25, color="#CCCCCC", linestyle="--", linewidth=0.6)
    axL.grid(False, axis="x")
    axL.set_axisbelow(True)
    axL.set_ylim(0, max(spam_vals.max(), good_vals.max()) * 1.2)

    # === Right axis: spam rate line ===
    axR = axL.twinx()
    rate_pct = pd.Series(rate_vals, index=months_all, dtype="float64") * 100.0
    mask = rate_pct.notna().to_numpy()

    # Solid black spam-rate line
    axR.plot(x[mask], rate_pct.values[mask],
             color="#000000", linewidth=2.4, marker="o",
             markersize=6.5, markerfacecolor="black",
             label="Spam rate", zorder=5)

    # === Regression (trend) line ===
    if len(x[mask]) > 2:
        coeffs = np.polyfit(x[mask], rate_pct.values[mask], deg=1)
        trend_line = np.polyval(coeffs, x[mask])
        label_corr = f"Linear trend (r={r_val:.3f}, p={p_val:.3f})" if r_val is not None else "Linear trend"
        axR.plot(x[mask], trend_line,
                 color="#4D4D4D", linestyle="--", linewidth=2.2,
                 label=label_corr, zorder=4)

    axR.set_ylabel("Spam rate [%]", fontsize=15, fontweight="bold")
    axR.tick_params(axis="y", labelsize=14)
    axR.set_ylim(30, 100)
    axR.yaxis.set_major_formatter(PercentFormatter(100))
    axR.grid(axis="y", alpha=0.15, linestyle="--")

    # === Milestone markers ===
    add_vertical_markers(axR, months_all)

    # === Legend (top-right, boxed) ===
    h1, l1 = axL.get_legend_handles_labels()
    h2, l2 = axR.get_legend_handles_labels()
    handles = h1 + h2
    labels = l1 + l2
    legend = axL.legend(handles, labels,
                        loc="upper right", fontsize=14,
                        frameon=True, framealpha=1.0,
                        edgecolor="#AAAAAA", fancybox=False,
                        borderpad=0.6)
    legend.get_frame().set_linewidth(0.8)

    # === Title ===
    # plt.title(title, fontsize=18, fontweight="bold", pad=14)

    # === Axes styling ===
    # --- Unified border (spines) styling ---
    for ax in [axL, axR] if 'axL' in locals() else [axH, axAI, axR]:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#222222")   # dark gray (not pure black)
            spine.set_linewidth(1.2)     # uniform border width



    plt.tight_layout(pad=1.2)
    plt.savefig(out_path, bbox_inches="tight", dpi=DPI)
    plt.close()
    print(f"✅ Saved: {out_path}")

# ========= MAIN =========
def main():
    spam = pd.read_csv(SPAM_CSV, on_bad_lines="skip", low_memory=False)
    for col in ("URL", "Timestamp", "Binary_Class", "Review_Type"):
        if col not in spam.columns:
            raise ValueError(f"Missing column in SPAM_CSV: {col}")

    spam["Binary_Class"] = spam["Binary_Class"].apply(normalize_binary_class)
    spam["Review_Type"] = spam["Review_Type"].apply(normalize_review_type)
    spam["Timestamp"] = pd.to_datetime(spam["Timestamp"], errors="coerce").dt.tz_localize(None)
    spam = spam[spam["Binary_Class"] == "review"].copy()
    spam["month"] = spam["Timestamp"].dt.to_period("M").dt.to_timestamp()

    if isinstance(MONTH_LOOKBACK, int) and MONTH_LOOKBACK > 0:
        cutoff = pd.Timestamp.today().to_period("M").to_timestamp() - pd.DateOffset(months=MONTH_LOOKBACK - 1)
        spam = spam[spam["month"] >= cutoff]

    cand = pd.read_csv(CANDIDATES_CSV, on_bad_lines="skip", low_memory=False)
    cand_url_col = next((c for c in ("URL", "url", "Url") if c in cand.columns), None)
    if not cand_url_col:
        raise ValueError("Missing URL column in candidates CSV")
    path_col = find_pathlike_column(cand)
    if not path_col:
        raise ValueError("No path-like column in candidates CSV")

    cand["engine"] = cand[path_col].apply(extract_engine_from_text)
    cand_small = cand[[cand_url_col, "engine"]].dropna().drop_duplicates()
    cand_small.rename(columns={cand_url_col: "URL"}, inplace=True)
    spam = spam.merge(cand_small, on="URL", how="left")
    spam["engine"] = spam["engine"].replace({"ddg": "duckduckgo"})

    def decide(series: pd.Series) -> str:
        s = series.astype(str).str.lower()
        if (s == "spam").any(): return "spam"
        if (s == "good").any(): return "good"
        return ""

    per_mon_url = (
        spam.groupby(["engine", "month", "URL"], as_index=False)
            .agg(final_label=("Review_Type", decide))
    )
    per_mon_url = per_mon_url[per_mon_url["final_label"] != ""].copy()

    agg = (
        per_mon_url.assign(is_spam=lambda d: (d["final_label"] == "spam").astype(int),
                           is_good=lambda d: (d["final_label"] == "good").astype(int))
        .groupby(["engine", "month"], as_index=False)
        .agg(spam=("is_spam", "sum"),
             good=("is_good", "sum"))
    )
    agg["total"] = agg["spam"] + agg["good"]
    agg["spam_rate"] = np.where(agg["total"] > 0, agg["spam"] / agg["total"], np.nan)

    engines = [e for e in ["bing", "duckduckgo", "startpage"] if e in agg["engine"].unique()]
    if not engines:
        engines = [e for e in agg["engine"].dropna().unique() if e != "unknown"]

    results_corr = []

    for eng in engines:
        sub = agg[agg["engine"] == eng].copy()
        sub = sub[sub["total"] >= MIN_MONTH_TOTAL_BY_ENGINE.get(eng, 0)]
        sub = sub.sort_values("month")
        if sub.empty:
            continue

        valid = sub["spam_rate"].notna()
        if valid.sum() > 2:
            month_numeric = pd.to_datetime(sub.loc[valid, "month"]).map(pd.Timestamp.toordinal)
            r_val, p_val = pearsonr(month_numeric, sub.loc[valid, "spam_rate"])
        else:
            r_val, p_val = np.nan, np.nan

        months_all = pd.DatetimeIndex(sub["month"])
        dual_axis_plot(
            months_all=months_all,
            good_vals=sub["good"].to_numpy(),
            spam_vals=sub["spam"].to_numpy(),
            rate_vals=sub["spam_rate"].to_numpy(),
            title=f"{eng.capitalize()} — Monthly Unique Reviews: Spam vs Good & Spam Rate",
            out_path=os.path.join(OUT_DIR, f"spam_good_rate_{eng}.png"),
            r_val=r_val,
            p_val=p_val
        )

        print(f"\n📈 {eng.capitalize()}: Pearson r = {r_val:.3f}, p = {p_val:.3e}")
        results_corr.append((eng, r_val, p_val))

    # Save correlation summary
    pd.DataFrame(results_corr, columns=["engine", "r_value", "p_value"]).to_csv(
        os.path.join(OUT_DIR, "spam_rate_trend_correlation.csv"), index=False
    )

    agg.sort_values(["engine", "month"]).to_csv(
        os.path.join(OUT_DIR, "per_engine_monthly_spam_counts.csv"), index=False
    )
    print("💾 Saved aggregated CSVs and correlation table")

if __name__ == "__main__":
    main()


