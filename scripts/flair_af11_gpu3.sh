#!/bin/bash
# User request 2026-09-12: T1 keeps the AF12 mask, FLAIR uses the AF11 mask; P10 recipe (baked in the runner), only MC-GS re-run.
cd /data0/congcong/code/GS/mcgs_af12_wt
CONFIG=configs/t1flair_v3_af12_hyw_flair_af11.yaml SAVE_VOLUMES=1 ./autoresearch_jyz_af12.sh ar9_P10_flair_af11_full4k 3 4000
