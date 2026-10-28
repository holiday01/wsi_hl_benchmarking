#!/usr/bin/env bash
# Download the 50 lesion-annotated CAMELYON17 training slides (10 per centre), their
# tumour masks and XML annotations from the public S3 bucket (no credentials needed),
# then verify every file against the bucket's checksums.md5.
#   bash scripts/data/download_camelyon17.sh [DEST] [PARALLEL]
# Resumable: files whose size already matches the bucket listing are skipped, partial
# files are resumed. Exit status is non-zero unless all 150 files verify.
set -u
DEST="${1:-/data2/wsi_hl_raw/camelyon17}"
PAR="${2:-6}"
B="https://camelyon-dataset.s3.amazonaws.com"
LIST="$(readlink -f "$(dirname "$0")/camelyon17_annotated50.txt")"
mkdir -p "$DEST/images" "$DEST/masks" "$DEST/annotations"
curl -sS -m 120 -o "$DEST/stages.csv" "$B/CAMELYON17/stages.csv"
curl -sS -m 120 -o "$DEST/checksums.md5" "$B/CAMELYON17/checksums.md5"
curl -sS -m 120 -o "$DEST/license.txt" "$B/CAMELYON17/license.txt"
# remote sizes (key<TAB>size) for the three prefixes
SIZES="$DEST/remote_sizes.tsv"; : > "$SIZES"
for pfx in images masks annotations; do
  curl -sS -m 120 "$B/?prefix=CAMELYON17/$pfx/&max-keys=1000" | sed 's/></>\n</g' \
    | grep "<Key>\|<Size>" | paste - - | sed 's/<Key>//;s/<\/Key>//;s/<Size>//;s/<\/Size>//' >> "$SIZES"
done

fetch() {  # fetch <relative key> <local path>
  local key="$1" out="$2"
  local want; want=$(awk -F'\t' -v k="CAMELYON17/$key" '$1==k{print $2}' "$SIZES")
  if [ -n "$want" ] && [ -f "$out" ] && [ "$(stat -c %s "$out")" = "$want" ]; then echo "[have] $key"; return 0; fi
  # resume if partial; -f fails on HTTP errors; retry transient failures
  curl -sS -f -L --retry 8 --retry-delay 10 -C - -o "$out" "$B/CAMELYON17/$key" \
    && echo "[ok]   $key" || echo "[FAIL] $key"
}
export -f fetch; export B SIZES

{
  while read -r s; do
    echo "annotations/$s.xml $DEST/annotations/$s.xml"
    echo "masks/${s}_mask.tif $DEST/masks/${s}_mask.tif"
    echo "images/$s.tif $DEST/images/$s.tif"
  done < "$LIST"
} | xargs -P "$PAR" -n 2 bash -c 'fetch "$0" "$1"'

echo "[verify] md5 against checksums.md5"
cd "$DEST" || exit 1
BAD=0; N=0
while read -r s; do
  for rel in "images/$s.tif" "masks/${s}_mask.tif" "annotations/$s.xml"; do
    N=$((N+1))
    want=$(grep -E "[[:space:]]\*?(\./)?(CAMELYON17/)?$rel\$" checksums.md5 | awk '{print $1}' | head -1)
    if [ -z "$want" ]; then echo "[no-checksum] $rel"; BAD=$((BAD+1)); continue; fi
    if [ ! -f "$rel" ]; then echo "[missing] $rel"; BAD=$((BAD+1)); continue; fi
    got=$(md5sum "$rel" | awk '{print $1}')
    [ "$want" == "$got" ] && echo "[md5 ok] $rel" || { echo "[MD5 MISMATCH] $rel"; BAD=$((BAD+1)); }
  done
done < "$LIST"
echo "[done] $(date -Iseconds) checked=$N mismatches=$BAD"
[ "$N" -eq 150 ] && [ "$BAD" -eq 0 ]
