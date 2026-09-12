#!/usr/bin/env bash
# Phase-3 Round 0, card 5: sequential 1500-it screens off P6 (see autoresearch_af16_p3.md).
cd "$(dirname "$0")/.."
./autoresearch_af16_p3.sh p3_r0_db4l1 5 1500 --set loss.wav_name=db4 --set loss.wav_levels=1 --set loss.lambda_wav_plain=0.12 --set loss.lambda_wav=0.10
./autoresearch_af16_p3.sh p3_r0_vtv002 5 1500 --lambda_vtv 0.02
