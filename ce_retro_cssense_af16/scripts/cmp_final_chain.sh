#!/usr/bin/env bash
# cmp_final_chain.sh -- the comparison-arm audit's wrap-up (2026-10-10): wait for the outstanding runs (or a deadline), pick
# each arm's winner by its own rule (cmp_select.py), add the MC-GS / GS-single of record (the _ar2 winners), build the
# confirmation topic and rerun the bookkeeping with PNG figures (confirm_cmp.sh, FIGS=1). Never overwrites: make_cmp_topic.py
# refuses an existing output directory.
#   setsid nohup bash scripts/cmp_final_chain.sh > logs/cmp/final_chain.log 2>&1 &
#   WAIT="dip_sup_b96_tv0.03_pre" ... overrides the list of score files to wait for; DEADLINE="YYYY-MM-DD HH:MM" overrides the +12 h default
set -u
N=/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois
S=/data0/congcong/code/GS/mcgs_ce_mask_wt/ce_retro_cssense_af16/scripts
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
OUT=${OUT:-/data0/congcong/code/GS/Results_20260908/CE_retro_cssense_af16_vdpois_cmp}
DEADLINE=${DEADLINE:-$(date -d "+12 hours" "+%F %T")}   # relative to the launch (2026-10-10 pause: an absolute default would skip the wait after a late resume)
WAIT=${WAIT:-"zsssdu_z2_b15_cg10_e400_pre zsssdu_z2_b15_cg10_e400_post zsssdu_z3_official_pre zsssdu_z3_official_post
      inr_i10_n512_sup6_tv0.03_pre inr_i10_n512_sup6_tv0.03_post inr_i9_n256_sup6_tv0.01_post dip_sup_b96_it12k_post
      dip_sup_b96_tv0.03_pre dip_sup_b96_tv0.03_post"}   # 04:55: the 20k DIP probe and the DIP TV 0.1 bracket were dropped (card collisions; optional)
cd $N
end=$(date -d "$DEADLINE" +%s)
while :; do
  miss=""; for k in $WAIT; do [ -f logs/$k.json ] || miss="$miss $k"; done
  [ -z "$miss" ] && { echo "$(date '+%F %T') all outstanding runs scored"; break; }
  [ $(date +%s) -ge $end ] && { echo "$(date '+%F %T') DEADLINE reached; still missing:$miss"; break; }
  sleep 120
done
env RETRO_TOPIC=$N PYTHONNOUSERSITE=1 PYTHONPATH= $PY $S/cmp_select.py > logs/cmp/select_final.txt 2>&1
cat logs/cmp/select_final.txt
REPL=$(grep '^REPLACE ' logs/cmp/select_final.txt | sed 's/^REPLACE //')
REPL="$REPL --replace gs_joint:pre=ar_i1_k30p90 --replace gs_joint:post=ar_i1_k30p90 --replace gs_single:pre=ar_s_pre_wf00v --replace gs_single:post=ar_s_post_wf00bv"
echo "$(date '+%F %T') building $OUT with: $REPL"
FIGS=1 bash $S/confirm_cmp.sh "$OUT" "$REPL" "final (cmp_select.py at $(date '+%F %T'); MC-GS / GS-single = the _ar2 winners)"
echo "$(date '+%F %T') FINAL_CHAIN_DONE"
