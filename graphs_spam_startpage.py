#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import pearsonr

# ========= CONFIG =========
SPAM_CSV_STARTPAGE = "/mnt/ceph/storage/data-tmp/current/yili5634/bert_predictions_binary_multiclass_v4.csv"
OUT_DIR = "./ai_human_plots_spam_startpage"
MONTH_LOOKBACK = 40

COLOR_SPAM = "#CC6677"
COLOR_GOOD = "#88CCEE"
COLOR_RATE = "#000000"
COLOR_TREND = "#4D4D4D"
DPI = 600

IMPORTANT_DATES = [
    ("ChatGPT launch", "2022-11-30"),
    ("Bing Chat", "2023-02-07"),
    ("GPT-4", "2023-03-14"),
]

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

def add_vertical_markers(ax_right, months_all: pd.DatetimeIndex):
    """
    Same styling as original (your preferred version),
    but with Bing Chat placed exactly between ChatGPT and GPT-4
    if its month is missing or outside the range.
    """
    if months_all.empty:
        return

    # Convert months to lookup positions
    idx_map = {pd.Timestamp(m).to_period("M").to_timestamp(): i
               for i, m in enumerate(months_all)}

    # Store positions for events
    pos = {}

    # First pass: try to locate ChatGPT + GPT-4 normally
    for label, d in IMPORTANT_DATES:
        ts = pd.to_datetime(d, errors="coerce")
        if pd.isna(ts):
            continue
        mstart = ts.to_period("M").to_timestamp()

        # If exact month exists, use exact index
        if mstart in idx_map:
            pos[label] = idx_map[mstart]
        else:
            # Find nearest month (fallback)
            diffs = [abs((m - mstart).days) for m in months_all]
            if diffs:
                pos[label] = int(np.argmin(diffs))

    # Second pass: compute Bing Chat midpoint if missing
    if "Bing Chat" in [lbl for lbl, _ in IMPORTANT_DATES]:
        # Get ChatGPT & GPT-4 positions if available
        cg = pos.get("ChatGPT launch", None)
        g4 = pos.get("GPT-4", None)

        if cg is not None and g4 is not None:
            pos["Bing Chat"] = (cg + g4) / 2.0  # midpoint
        # else: keep whatever fallback was found

    # Draw markers using your preferred styling
    y_top = ax_right.get_ylim()[1]

    for label, x in pos.items():
        ax_right.axvline(
            x,
            color="#666666",
            linestyle="--",
            linewidth=1.1,
            alpha=0.9,
            zorder=2
        )

        # consistent vertical placement
        ax_right.text(
            x - 0.2,
            y_top * 0.97,
            label,
            rotation=90,
            va="top",
            ha="right",
            fontsize=13,
            color="#444444",
            backgroundcolor="white",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.3),
            zorder=6
        )


# ========= PLOT =========
def dual_axis_plot(months_all, good_vals, spam_vals, rate_vals, title, out_path, r_val=None, p_val=None):
    import matplotlib.ticker as mticker

    fig, axL = plt.subplots(figsize=(16, 6.5), dpi=DPI)
    x = np.arange(len(months_all), dtype=float)

    # --- Bars (left axis) ---
    w = 0.4
    axL.bar(x - w/2, good_vals, width=w, color=COLOR_GOOD, label="Good", alpha=0.95, zorder=1)
    axL.bar(x + w/2, spam_vals, width=w, color=COLOR_SPAM, label="Spam", alpha=0.95, zorder=1)

    labels = pd.Index(months_all).strftime("%Y-%m").tolist()
    step = max(1, int(np.ceil(len(labels) / 12)))

    axL.set_xticks(x)
    axL.set_xticklabels(
        [lab if i % step == 0 else "" for i, lab in enumerate(labels)],
        rotation=45, fontsize=14
    )
    axL.set_ylabel("Unique Review URLs", fontsize=16, fontweight="bold")
    axL.set_xlabel("Month", fontsize=16, fontweight="bold")
    axL.tick_params(axis="y", labelsize=14)
    axL.grid(axis="y", alpha=0.3, color="#D9D9D9", linewidth=0.6)
    axL.grid(False, axis="x")
    axL.set_axisbelow(True)

    axL.set_ylim(0, 5000)
    axL.yaxis.set_major_locator(mticker.MultipleLocator(1000))

    # --- Right axis (Spam rate) ---
    axR = axL.twinx()
    rate_pct = pd.Series(rate_vals, index=months_all, dtype="float64") * 100.0
    mask = rate_pct.notna().to_numpy()

    # Main spam rate line
    axR.plot(
        x[mask], rate_pct.values[mask],
        color=COLOR_RATE, linewidth=2.4, marker="o",
        markersize=6.5, markerfacecolor="black",
        label="Spam rate", zorder=5
    )
    axR.set_ylabel("Spam rate [%]", fontsize=16, fontweight="bold")
    axR.tick_params(axis="y", labelsize=14)
    axR.set_ylim(30, 100)
    axR.yaxis.set_major_formatter(PercentFormatter(100))

    # --- Add trend line (linear fit over spam rate) ---
    valid = mask & ~np.isnan(rate_pct.values)
    if valid.sum() > 2:
        coeffs = np.polyfit(x[valid], rate_pct.values[valid], deg=1)
        trend_line = np.polyval(coeffs, x[valid])
        label_corr = f"Linear trend (r={r_val:.3f}, p={p_val:.3f})" if r_val is not None else "Linear trend"
        axR.plot(x[valid], trend_line, color=COLOR_TREND, linestyle="--",
                 linewidth=2.2, label=label_corr, zorder=3)

    # --- Milestones ---
    add_vertical_markers(axR, months_all)

    # --- Legend ---
    h1, l1 = axL.get_legend_handles_labels()
    h2, l2 = axR.get_legend_handles_labels()
    legend = axL.legend(
        h1 + h2, l1 + l2,
        loc="upper right", fontsize=14,
        frameon=True, framealpha=1.0,
        edgecolor="#AAAAAA", fancybox=False,
        borderpad=0.6
    )
    legend.get_frame().set_linewidth(0.8)

    # --- Styling ---
    # plt.title(title, fontsize=18, fontweight="bold", pad=14)
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
    spam = pd.read_csv(SPAM_CSV_STARTPAGE, on_bad_lines="skip", low_memory=False)
    spam["engine"] = "startpage"

    for col in ("URL", "Timestamp", "Binary_Class", "Review_Type"):
        if col not in spam.columns:
            raise ValueError(f"Missing column in SPAM_CSV: {col}")

    spam["Binary_Class"] = spam["Binary_Class"].apply(normalize_binary_class)
    spam["Review_Type"] = spam["Review_Type"].apply(normalize_review_type)
    spam["Timestamp"] = pd.to_datetime(spam["Timestamp"], errors="coerce")
    spam = spam[spam["Binary_Class"] == "review"].copy()
    spam["month"] = spam["Timestamp"].dt.to_period("M").dt.to_timestamp()

    if isinstance(MONTH_LOOKBACK, int) and MONTH_LOOKBACK > 0:
        month_start = pd.Timestamp.today().to_period("M").to_timestamp()
        cutoff = month_start - pd.DateOffset(months=MONTH_LOOKBACK - 1)
        spam = spam[spam["month"] >= cutoff]

    print(f"Total rows: {len(spam)}")
    print("Review type distribution:")
    print(spam["Review_Type"].value_counts())
    print("Unique URLs:", spam["URL"].nunique())

    def decide(series: pd.Series) -> str:
        s = series.astype(str).str.lower()
        if (s == "spam").any():
            return "spam"
        if (s == "good").any():
            return "good"
        return ""

    per_mon_url = (
        spam.groupby(["month", "URL"], as_index=False)
            .agg(final_label=("Review_Type", decide))
    )
    per_mon_url = per_mon_url[per_mon_url["final_label"] != ""].copy()

    agg = (
        per_mon_url.assign(
            is_spam=lambda d: (d["final_label"] == "spam").astype(int),
            is_good=lambda d: (d["final_label"] == "good").astype(int)
        )
        .groupby(["month"], as_index=False)
        .agg(spam=("is_spam", "sum"), good=("is_good", "sum"))
    )

    agg["total"] = agg["spam"] + agg["good"]
    agg["spam_rate"] = np.where(agg["total"] > 0, agg["spam"] / agg["total"], np.nan)
    agg = agg.sort_values("month")

    # --- Compute correlation between month and spam rate ---
    valid = agg["spam_rate"].notna()
    if valid.sum() > 2:
        month_numeric = pd.to_datetime(agg.loc[valid, "month"]).map(pd.Timestamp.toordinal)
        r_val, p_val = pearsonr(month_numeric, agg.loc[valid, "spam_rate"])
    else:
        r_val, p_val = np.nan, np.nan

    # --- Plot ---
    months_all = pd.DatetimeIndex(agg["month"])
    dual_axis_plot(
        months_all=months_all,
        good_vals=agg["good"].to_numpy(),
        spam_vals=agg["spam"].to_numpy(),
        rate_vals=agg["spam_rate"].to_numpy(),
        title="Startpage — Monthly Unique Reviews: Spam vs Good & Spam Rate",
        out_path=os.path.join(OUT_DIR, "spam_good_rate_startpage.png"),
        r_val=r_val,
        p_val=p_val
    )

    print(f"\n📈 Pearson correlation between month and spam rate: r = {r_val:.3f}, p = {p_val:.3f}")
    agg.to_csv(os.path.join(OUT_DIR, "startpage_monthly_spam_counts.csv"), index=False)
    print("💾 Saved aggregated CSV")

if __name__ == "__main__":
    main()

