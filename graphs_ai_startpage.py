#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import pearsonr

# ========= CONFIG =========
INPUT_CSV_STARTPAGE = "/mnt/ceph/storage/data-tmp/current/yili5634/classified_results_v3.csv"
OUT_DIR = "./ai_human_plots_ai_startpage"
MONTH_LOOKBACK = 37

IMPORTANT_DATES = [
    ("ChatGPT launch", "2022-11-30"),
    ("Bing Chat", "2023-02-07"),
    ("GPT-4", "2023-03-14"),
]

# --- Colours to match spam plots ---
COLOR_HUMAN = "#88CCEE"   # blue
COLOR_AI    = "#CC6677"   # red
COLOR_AAR   = "#000000"   # black line
COLOR_TREND = "#4D4D4D"   # grey dashed
DPI = 600

os.makedirs(OUT_DIR, exist_ok=True)


# ---------- Helpers ----------
def normalize_ai_label(x: str) -> str:
    s = str(x).strip().lower()
    if s in {"ai", "ai detected", "ai_generated", "ai-generated"}:
        return "ai"
    if s in {"human", "human-written", "human written"}:
        return "human"
    return s


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



# ---------- Plot ----------
def dual_left_plus_right_plot(months, human_vals, ai_vals, aar_vals, title, out_path, r_val=None, p_val=None):
    x = np.arange(len(months), dtype=float)
    labels = pd.Index(months).strftime("%Y-%m").tolist()
    step = max(1, int(np.ceil(len(labels) / 12)))

    fig, axH = plt.subplots(figsize=(16, 6.5), dpi=DPI)
    fig.patch.set_facecolor("white")
    axH.set_facecolor("white")

    axAI = axH.twinx()
    axAI.spines["left"].set_position(("axes", -0.08))
    axAI.spines["left"].set_visible(True)
    axAI.yaxis.set_label_position("left")
    axAI.yaxis.set_ticks_position("left")

    axR = axH.twinx()

    width = 0.4

    # --- Bars ---
    axH.bar(x + width / 2, human_vals, width=width, color=COLOR_HUMAN,
            label="Human", align="center", alpha=0.95, zorder=2)
    axAI.bar(x - width / 2, ai_vals, width=width, color=COLOR_AI,
             label="AI", align="center", alpha=0.95, zorder=2)

    # --- AAR line ---
    aar_pct = np.asarray(aar_vals) * 100.0
    axR.plot(x, aar_pct, color=COLOR_AAR, linewidth=2.3,
             marker="o", markersize=5, markerfacecolor="black",
             label="AAR", zorder=4)

    # --- Linear trend line ---
    valid = ~np.isnan(aar_pct)
    if valid.sum() > 2:
        coeffs = np.polyfit(x[valid], aar_pct[valid], deg=1)
        trend_line = np.polyval(coeffs, x[valid])
        label_corr = f"Linear trend (r={r_val:.3f},p={int(p_val == 0) * 0})" if r_val is not None else "Linear trend"
        axR.plot(x[valid], trend_line, color=COLOR_TREND, linestyle="--",
                 linewidth=2.2, label=label_corr, zorder=3)

    # --- X-axis ---
    axH.set_xticks(x)
    axH.set_xticklabels([lab if i % step == 0 else "" for i, lab in enumerate(labels)],
                        rotation=45, fontsize=13, color="black")
    axH.set_xlabel("Month", fontsize=14, fontweight="bold", color="black")

    # --- Labels ---
    axH.set_ylabel("Human URL Count", fontsize=14, fontweight="bold", color="#1F77B4")
    axAI.set_ylabel("AI URL Count", fontsize=14, fontweight="bold", color="#B22222")
    axR.set_ylabel("AI Attribution Rate (AAR) [%]", fontsize=14, fontweight="bold", color="black")

    # --- Headroom adjustment ---
    max_left = max(np.nanmax(human_vals), np.nanmax(ai_vals))
    axH.set_ylim(0, max_left * 1.35)
    axAI.set_ylim(0, 900)

    # --- Right axis headspace ---
    max_right = np.nanmax(aar_pct)
    axR.set_ylim(0, 3.0)
    axR.yaxis.set_major_formatter(PercentFormatter())

    # --- Adjust figure margins ---
    fig.subplots_adjust(top=0.83)  # Adds extra space above legend


    # --- Grid & styling ---
    for ax in [axH, axAI, axR]:
        ax.tick_params(axis="both", colors="black", labelsize=12, width=1.3)
        for spine in ax.spines.values():
            spine.set_color("#222222")
            spine.set_linewidth(1.2)
    axH.grid(True, axis="y", color="#E0E0E0", linewidth=0.7, alpha=0.6)
    axH.grid(False, axis="x")

    add_vertical_markers(axR, months)

    # --- Legend ---
    h1, l1 = axH.get_legend_handles_labels()
    h2, l2 = axAI.get_legend_handles_labels()
    h3, l3 = axR.get_legend_handles_labels()
    legend = axH.legend(h1 + h2 + h3, l1 + l2 + l3, loc="upper right",
                        fontsize=13, frameon=True, framealpha=1.0,
                        edgecolor="#AAAAAA", fancybox=False, borderpad=0.4)
    legend.get_frame().set_linewidth(0.8)

    # plt.title(title, fontsize=18, fontweight="bold", color="black", pad=10)
    plt.tight_layout(pad=1.5)
    plt.savefig(out_path, bbox_inches="tight", dpi=DPI)
    plt.close()
    print(f"✅ Saved: {out_path}")


# ---------- MAIN ----------
def main():
    df = pd.read_csv(INPUT_CSV_STARTPAGE, on_bad_lines="skip", low_memory=False)
    print(f"Loaded {len(df)} rows from Startpage CSV")

    # --- Rename and normalize ---
    rename = {"URL": "url", "Timestamp": "timestamp", "AI_Label": "ai_label", "ai label": "ai_label"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    for c in ["url", "timestamp", "ai_label"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["month"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()
    df["ai_label"] = df["ai_label"].apply(normalize_ai_label)
    df = df[df["ai_label"].isin(["ai", "human"])].copy()

    if isinstance(MONTH_LOOKBACK, int) and MONTH_LOOKBACK > 0:
        cutoff = pd.Timestamp.today().to_period("M").to_timestamp() - pd.DateOffset(months=MONTH_LOOKBACK - 1)
        df = df[df["month"] >= cutoff]

    print("AI/Human label distribution:")
    print(df["ai_label"].value_counts())

    agg = (
        df.assign(
            is_ai=lambda d: (d["ai_label"] == "ai").astype(int),
            is_human=lambda d: (d["ai_label"] == "human").astype(int)
        )
        .groupby("month", as_index=False)
        .agg(ai=("is_ai", "sum"), human=("is_human", "sum"))
    )
    agg["total"] = agg["ai"] + agg["human"]
    agg["aar"] = np.where(agg["total"] > 0, agg["ai"] / agg["total"], np.nan)
    agg = agg.sort_values("month")

    # --- Pearson correlation ---
    valid = agg["aar"].notna()
    if valid.sum() > 2:
        month_numeric = pd.to_datetime(agg.loc[valid, "month"]).map(pd.Timestamp.toordinal)
        r_val, p_val = pearsonr(month_numeric, agg.loc[valid, "aar"])
    else:
        r_val, p_val = np.nan, np.nan

    # --- Plot ---
    months = pd.DatetimeIndex(agg["month"]).sort_values().unique()
    dual_left_plus_right_plot(
        months=months,
        human_vals=agg["human"].to_numpy(),
        ai_vals=agg["ai"].to_numpy(),
        aar_vals=agg["aar"].to_numpy(),
        title="Startpage — Monthly AI vs Human Reviews (Dual Left Axes) + AAR",
        out_path=os.path.join(OUT_DIR, "ai_vs_human_startpage.png"),
        r_val=r_val,
        p_val=p_val
    )

    print(f"\n📈 Pearson correlation between month and AAR: r = {r_val:.3f}, p = {p_val:.3f}")
    agg.to_csv(os.path.join(OUT_DIR, "startpage_monthly_ai_counts.csv"), index=False)
    print("💾 Saved aggregated CSV")

if __name__ == "__main__":
    main()


