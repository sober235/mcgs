#!/usr/bin/env bash
# archive_and_relabel_figs.sh -- the 180/124/76 inset48 figures of record of AF12/16/20 carry the
# superseded FLAIR acceleration label (against the contrast's own shutter: 9.74 / 12.49 / 15.13).
# User 2026-09-15: keep the old files, then re-archive corrected ones. So: move each pair (PNG + PDF,
# both contrasts) into archive_aflabel_pre20260915/ -- in the topic's figures/ and in the served
# ~/figs mirror alike, never deleting -- and re-render it under its canonical name with the same
# arms it had and the corrected title (AF_vs_full_rect). L1-ESPIRiT stays in these: they are the
# historical figures; the current deliverable is the _spirit set. Verification afterwards: T1 must
# come out pixel-identical to its archived copy (its label never changed), FLAIR may differ only in
# the title band.
set -u
R=/data0/congcong/code/GS/Results_20260908
F=/home/congcongliu/figs
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
C="--slices 180 124 76 --inset 48 --inset_corner tr bl bl --mask_error --gray_cbar --cbar_planes 0 --tra_rot 1 --err_vmax 0.1 --pdf"
ARCH=archive_aflabel_pre20260915

archive() {  # af  tag
  local d=$1 tag=$2 t
  case $d in af12) t=T1FLAIR_AF12_v3_jiangyizhou;; af16) t=T1FLAIR_AF16_v3_jiangyizhou;; af20) t=T1FLAIR_AF20_v3_jiangyizhou;; esac
  mkdir -p "$R/$t/figures/$ARCH" "$F/t1flair_v3_$d/$ARCH"
  for c in T1 FLAIR; do for e in png pdf; do
    f="t1flair_v3_${d}_${c}_row3planes_compact_180_124_76_inset48_headerr${tag}.$e"
    [ -e "$R/$t/figures/$f" ] && mv -n "$R/$t/figures/$f" "$R/$t/figures/$ARCH/"
    [ -e "$F/t1flair_v3_$d/$f" ] && mv -n "$F/t1flair_v3_$d/$f" "$F/t1flair_v3_$d/$ARCH/"
  done; done
  echo "archived $d $tag"
}

render() {  # topic  script  args...
  local t=$1 s=$2; shift 2
  cd "$R/$t" || return 1
  for c in T1 FLAIR; do PYTHONNOUSERSITE=1 PYTHONPATH= $PY "scripts/$s" $C --contrast $c "$@"; done
}

af12() {
  archive af12 _p14; archive af12 _p14_mcinr
  render T1FLAIR_AF12_v3_jiangyizhou fig_three_planes_inset.py --mcgs_key gs_joint_p14 --single_key gs_single_p14 --tag _p14 --no_mcinr
  render T1FLAIR_AF12_v3_jiangyizhou fig_three_planes_inset.py --mcgs_key gs_joint_p14 --single_key gs_single_p14 --tag _p14_mcinr
  echo "TOPIC_DONE af12"
}
af16() {
  archive af16 _p3; archive af16 _p3_mcinr
  render T1FLAIR_AF16_v3_jiangyizhou fig_three_planes_compact.py --mcgs_key gs_joint_p3 --single_key gs_single_p3 --tag _p3 --no_mcinr
  render T1FLAIR_AF16_v3_jiangyizhou fig_three_planes_compact.py --mcgs_key gs_joint_p3 --single_key gs_single_p3 --tag _p3_mcinr
  echo "TOPIC_DONE af16"
}
af20() {
  archive af20 _p3; archive af20 _p3_mcinr
  render T1FLAIR_AF20_v3_jiangyizhou fig_three_planes_compact.py --mcgs_key gs_joint_p3 --single_key gs_single_p3 --tag _p3
  render T1FLAIR_AF20_v3_jiangyizhou fig_three_planes_compact.py --mcgs_key gs_joint_p3 --single_key gs_single_p3 --tag _p3_mcinr --with_mcinr
  echo "TOPIC_DONE af20"
}

L=$R/T1FLAIR_AF16_v3_jiangyizhou/logs/spirit
af12 > $L/relabel_archive_af12.log 2>&1 &
af16 > $L/relabel_archive_af16.log 2>&1 &
af20 > $L/relabel_archive_af20.log 2>&1 &
wait
echo "ARCHIVE_RELABEL_DONE"
