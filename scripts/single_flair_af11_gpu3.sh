#!/bin/bash
# GS-single FLAIR (P10 recipe) with the FLAIR mask at AF 11, to compare like-for-like with ar9_P10_flair_af11_full4k.
cd /data0/congcong/code/GS/mcgs_af12_wt
CONFIG=configs/t1flair_v3_af12_hyw_singleFLAIR_af11.yaml SAVE_VOLUMES=1 ./autoresearch_jyz_af12.sh ar9_P10_singleFLAIR_af11_full4k 3 4000
