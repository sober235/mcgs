#!/usr/bin/env bash
# rerender_fs22.sh -- re-render the eight figures of record with 22 pt first-row method names (user 2026-09-16 evening),
# collect them into this delivery folder (manifest refreshed) and refresh the short-name served copies af<AF>_<c>.png.
set -u
H=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF12_16_20_24_figures_hires
FIGS=/home/congcongliu/figs/t1flair_v3_af_hires
echo "start $(date -Is)"
TITLE_FS=22 bash "$H/scripts/render_hires_final.sh"
bash "$H/scripts/collect_delivery.sh"
cd "$FIGS" && for af in 12 16 20 24; do for c in T1 FLAIR; do cp -f t1flair_v3_af${af}_${c}_row3planes_compact_180_124_76_inset48_headerr_*_mcinr_spirit_zf_caps.png "af${af}_${c}.png"; done; done
echo "RERENDER_FS22_DONE $(date -Is)"
