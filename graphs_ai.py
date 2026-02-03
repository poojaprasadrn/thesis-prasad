#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import pearsonr

# ========= CONFIG =========
INPUT_CSV_MAIN = "/mnt/ceph/storage/data-tmp/current/yili5634/classified_results_v1.csv"
INPUT_CSV_STARTPAGE = "/mnt/ceph/storage/data-tmp/current/yili5634/classified_results_v3.csv"
OUT_DIR = "./ai_human_plots_ai"
MONTH_LOOKBACK = 40

MIN_MONTH_TOTAL_BY_ENGINE = {"bing": 50, "duckduckgo": 50, "startpage": 50}

IMPORTANT_DATES = [
    ("ChatGPT launch", "2022-11-30"),
    ("Bing Chat", "2023-02-07"),
    ("GPT-4", "2023-03-14"),
]

# --- Colours identical to Startpage ---
COLOR_HUMAN = "#88CCEE"
COLOR_AI = "#CC6677"
COLOR_AAR = "#000000"
COLOR_TREND = "#4D4D4D"
DPI = 600

AI_YLIM = {"bing": 2500, "duckduckgo": 1100, "startpage": 900}

os.makedirs(OUT_DIR, exist_ok=True)

# ========= HELPERS =========
def extract_engine_from_path(path: str) -> str:
    p = str(path).lower()
    if "duckduckgo" in p or "ddg" in p:
        return "duckduckgo"
    elif "bing" in p:
        return "bing"
    elif "startpage" in p:
        return "startpage"
    m = re.search(r"/?20\d{2}-\d{2}-\d{2}-([a-z0-9\-]+)/", p)
    if m:
        return m.group(1)
    return "unknown"


def normalize_ai_label(x: str) -> str:
    s = str(x).strip().lower()
    if s in {"ai", "ai detected", "ai_generated", "ai-generated"}:
        return "ai"
    if s in {"human", "human-written", "human written"}:
        return "human"
    return s


def decide_final(series: pd.Series) -> str:
    s = series.astype(str).str.lower()
    if (s == "ai").any():
        return "ai"
    if (s == "human").any():
        return "human"
    return ""



def add_vertical_markers(ax_right, months_all: pd.DatetimeIndex):
    """
    Original marker styling (exactly as in your second function).
    Draw markers ONLY when that month actually exists in months_all.
    No midpoint shifting or nearest-match fallback.
    """
    if months_all.empty:
        return

    for label, d in IMPORTANT_DATES:
        ts = pd.to_datetime(d, errors="coerce")
        if pd.isna(ts):
            continue

        # Convert to first day of month
        mstart = ts.to_period("M").to_timestamp()

        # Skip if month outside plotted range
        if not (months_all.min() <= mstart <= months_all.max()):
            continue

        # Skip if this exact month is NOT in the data
        idx = np.where(months_all == mstart)[0]
        if len(idx) == 0:
            continue

        x = int(idx[0])
        y_top = ax_right.get_ylim()[1]

        # Vertical dashed line (your exact style)
        ax_right.axvline(
            x,
            color="#666666",
            linestyle="--",
            linewidth=1.1,
            alpha=0.9,
            zorder=2
        )

        # Label text (your exact style)
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
def dual_left_plus_right_plot(
    months, human_vals, ai_vals, aar_vals, title, out_path,
    r_val=None, p_val=None, ai_ylim=None
):
    x = np.arange(len(months), dtype=float)
    labels = pd.Index(months).strftime("%Y-%m").tolist()
    step = max(1, int(np.ceil(len(labels) / 12)))

    fig, axH = plt.subplots(figsize=(16, 6.5), dpi=DPI)
    fig.patch.set_facecolor("white")
    axH.set_facecolor("white")

    # --- Twin axes setup ---
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

    # --- Linear trend line with (r,p) ---
    valid = ~np.isnan(aar_pct)
    if valid.sum() > 2:
        coeffs = np.polyfit(x[valid], aar_pct[valid], deg=1)
        trend_line = np.polyval(coeffs, x[valid])

        label_corr = f"Linear trend (r={r_val:.3f}, p={int(p_val == 0) * 0})" if r_val is not None else "Linear trend"

        axR.plot(x[valid], trend_line, color=COLOR_TREND, linestyle="--",
                 linewidth=2.2, label=label_corr, zorder=3)

    # --- X-axis ---
    axH.set_xticks(x)
    axH.set_xticklabels(
        [lab if i % step == 0 else "" for i, lab in enumerate(labels)],
        rotation=45, fontsize=13, color="black"
    )
    axH.set_xlabel("Month", fontsize=14, fontweight="bold", color="black")

    # --- Labels ---
    axH.set_ylabel("Human URL Count", fontsize=14, fontweight="bold", color="#1F77B4")
    axAI.set_ylabel("AI URL Count", fontsize=14, fontweight="bold", color="#B22222")
    axR.set_ylabel("AI Attribution Rate (AAR) [%]", fontsize=14, fontweight="bold", color="black")

    # --- Headroom adjustments ---
    max_human = np.nanmax(human_vals) if np.nanmax(human_vals) > 0 else 1
    axH.set_ylim(0, max_human * 1.35)
    if ai_ylim is not None:
        axAI.set_ylim(0, ai_ylim)
    else:
        max_ai = np.nanmax(ai_vals) if np.nanmax(ai_vals) > 0 else 1
        axAI.set_ylim(0, max_ai * 1.35)
    axR.set_ylim(0, 3.0)
    axR.yaxis.set_major_formatter(PercentFormatter())

    fig.subplots_adjust(top=0.83)

    # --- Grid and styling ---
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
    legend = axH.legend(
        h1 + h2 + h3, l1 + l2 + l3,
        loc="upper right",
        fontsize=13, frameon=True, framealpha=1.0,
        edgecolor="#AAAAAA", fancybox=False, borderpad=0.4,
    )
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout(pad=1.5)
    plt.savefig(out_path, bbox_inches="tight", dpi=DPI)
    plt.close()
    print(f"✅ Saved: {out_path}")


# ========= MAIN =========
def main():
    df_main = pd.read_csv(INPUT_CSV_MAIN, on_bad_lines="skip", low_memory=False)
    df_startpage = pd.read_csv(INPUT_CSV_STARTPAGE, on_bad_lines="skip", low_memory=False)
    df = pd.concat([df_main, df_startpage], ignore_index=True)
    print(f"Loaded {len(df)} rows total")

    rename = {"URL": "url", "Timestamp": "timestamp", "Path": "path",
              "AI_Label": "ai_label", "ai label": "ai_label"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    for req in ["url", "timestamp", "path", "ai_label"]:
        if req not in df.columns:
            raise ValueError(f"Missing column: {req}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["month"] = df["timestamp"].dt.to_period("M").dt.to_timestamp()
    df["engine"] = df["path"].apply(extract_engine_from_path)
    df["ai_label"] = df["ai_label"].apply(normalize_ai_label)
    df = df[df["ai_label"].isin({"ai", "human"})].copy()

    if isinstance(MONTH_LOOKBACK, int) and MONTH_LOOKBACK > 0:
        cutoff = pd.Timestamp.today().to_period("M").to_timestamp() - pd.DateOffset(months=MONTH_LOOKBACK - 1)
        df = df[df["month"] >= cutoff]

    per_url = (
        df.groupby(["engine", "month", "url"], as_index=False)
          .agg(final=("ai_label", decide_final))
    )
    per_url = per_url[per_url["final"] != ""].copy()

    agg = (
        per_url.assign(
            is_ai=lambda d: (d["final"] == "ai").astype(int),
            is_human=lambda d: (d["final"] == "human").astype(int)
        )
        .groupby(["engine", "month"], as_index=False)
        .agg(ai=("is_ai", "sum"), human=("is_human", "sum"))
    )
    agg["total"] = agg["ai"] + agg["human"]
    agg["aar"] = np.where(agg["total"] > 0, agg["ai"] / agg["total"], np.nan)

    # --- Plot each engine ---
    for eng in ["bing", "duckduckgo", "startpage"]:
        if eng not in agg["engine"].unique():
            continue
        sub = agg[agg["engine"] == eng].copy()
        sub = sub[sub["total"] >= MIN_MONTH_TOTAL_BY_ENGINE.get(eng, 0)]
        sub = sub.sort_values("month")
        if sub.empty:
            continue

        # Pearson correlation (month vs AAR %)
        valid = sub["aar"].notna()
        if valid.sum() > 2:
            month_numeric = sub.loc[valid, "month"].astype(np.int64) // 10**9
            r_val, p_val = pearsonr(month_numeric, sub.loc[valid, "aar"])
        else:
            r_val, p_val = np.nan, np.nan

        print(f"[{eng}] r = {r_val:.3f}, p = {p_val:.3f}")

        months = pd.DatetimeIndex(sub["month"])
        dual_left_plus_right_plot(
            months=months,
            human_vals=sub["human"].to_numpy(),
            ai_vals=sub["ai"].to_numpy(),
            aar_vals=sub["aar"].to_numpy(),
            title=f"{eng.capitalize()} — Monthly Unique URLs: Human vs AI & AAR",
            out_path=os.path.join(OUT_DIR, f"ai_vs_human_{eng}.png"),
            r_val=r_val, p_val=p_val,
            ai_ylim=AI_YLIM.get(eng, 900)
        )

    agg.sort_values(["engine", "month"]).to_csv(
        os.path.join(OUT_DIR, "per_engine_monthly_ai_counts.csv"), index=False
    )
    print("💾 Saved CSV")


if __name__ == "__main__":
    main()

