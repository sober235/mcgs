#!/usr/bin/env bash
# Phase-3 Round 0, card 3: sequential 1500-it screens off P6 (see autoresearch_af16_p3.md).
cd "$(dirname "$0")/.."
./autoresearch_af16_p3.sh p3_r0_vtv0005 3 1500 --lambda_vtv 0.005
./autoresearch_af16_p3.sh p3_r0_edge0005 3 1500 --lambda_edge 0.005
./autoresearch_af16_p3.sh p3_r0_ptv002 3 1500 --lambda_ptv 0.02
