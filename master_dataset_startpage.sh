# edit these:
ROOT="/mnt/ceph/storage/data-in-progress/data-research/web-search/affiliate-serp-crawls"
OUT_DIR="/mnt/ceph/storage/data-tmp/current/yili5634"
V1="/mnt/ceph/storage/data-tmp/current/yili5634/startpage_10_per_date_v1.csv"

# pick your 6 target dates (example: the missing ones you listed earlier)
DATES=(2023-06-06 2023-08-23 2023-10-02 2024-07-26 2024-01-11 2024-03-08)

# how many URLs per date (cap)
K=100000

mkdir -p "$OUT_DIR/new_dates"
pids=()

for d in "${DATES[@]}"; do
  OUT="$OUT_DIR/new_dates/extract_$d.csv"
  echo "[spawn] $d -> $OUT"
  python -u master_dataset_startpage.py \
    --root "$ROOT" \
    --out "$OUT" \
    --per-date-k "$K" \
    --resume-csv "$V1" \
    --only-dates "$d" \
    >> "$OUT_DIR/new_dates/extract_$d.log" 2>&1 &
  pids+=($!)
done

# wait for all 6
for pid in "${pids[@]}"; do wait "$pid"; done

# merge into your v2 file (keep header once)
V2="$OUT_DIR/startpage_10_per_date_v2.csv"
TMP="$V2.new"
HEADER_SRC="${OUT_DIR}/new_dates/extract_${DATES[0]}.csv"

if [[ -f "$HEADER_SRC" ]]; then
  head -n 1 "$HEADER_SRC" > "$TMP"
  for f in "$OUT_DIR"/new_dates/extract_*.csv; do
    if [[ -f "$f" && $(wc -l < "$f") -gt 1 ]]; then
      tail -n +2 "$f" >> "$TMP"
    fi
  done
  mv -f "$TMP" "$V2"
  echo "Merged → $V2"
else
  echo "No outputs to merge."
fi
