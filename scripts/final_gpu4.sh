#!/bin/bash
# Card 4: the single-contrast arms of the final recipe P10 (rule 5: run and reported, not optimised).
cd /data0/congcong/code/GS/mcgs_af12_wt
CONFIG=configs/t1flair_v3_af12_hyw_singleT1.yaml SAVE_VOLUMES=1 ./autoresearch_jyz_af12.sh ar8_P10_singleT1_full4k 4 4000
CONFIG=configs/t1flair_v3_af12_hyw_singleFLAIR.yaml SAVE_VOLUMES=1 ./autoresearch_jyz_af12.sh ar8_P10_singleFLAIR_full4k 4 4000
