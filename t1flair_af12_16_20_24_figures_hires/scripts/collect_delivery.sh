#!/usr/bin/env bash
# collect_delivery.sh -- gather the four AF topics' high-resolution figure of record into one
# delivery folder (copies, not links, so the folder can be moved off the server as a unit) and
# mirror the PNGs into a single served folder for browsing.
set -eu
R=/data0/congcong/code/GS/Results_20260908
D=$R/T1FLAIR_AF12_16_20_24_figures_hires
FIGS=/home/congcongliu/figs/t1flair_v3_af_hires
mkdir -p "$D/png" "$D/pdf" "$FIGS"

n=0
for a in 12:p14 16:p3 20:p3 24:p3; do
  af=${a%%:*}; p=${a##*:}
  for c in T1 FLAIR; do
    base="t1flair_v3_af${af}_${c}_row3planes_compact_180_124_76_inset48_headerr_${p}_mcinr_spirit_zf_caps"
    src=$R/T1FLAIR_AF${af}_v3_jiangyizhou/figures/$base
    [ -f "$src.png" ] || { echo "MISSING $src.png"; exit 1; }
    [ -f "$src.pdf" ] || { echo "MISSING $src.pdf"; exit 1; }
    cp -f "$src.png" "$D/png/"; cp -f "$src.pdf" "$D/pdf/"; cp -f "$src.png" "$FIGS/"
    n=$((n+1))
  done
done
( cd "$D" && md5sum png/*.png pdf/*.pdf > manifest.md5 )
echo "collected $n figure pairs into $D"
du -sh "$D"
