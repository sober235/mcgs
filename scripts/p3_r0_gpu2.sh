#!/usr/bin/env bash
# Phase-3 Round 0, card 2: sequential 1500-it screens off P6 (see autoresearch_af16_p3.md).
cd "$(dirname "$0")/.."
./autoresearch_af16_p3.sh p3_r0_control 2 1500
./autoresearch_af16_p3.sh p3_r0_edge0001 2 1500 --lambda_edge 0.001
./autoresearch_af16_p3.sh p3_r0_priv03 2 1500 --set gaussian.num_private_points=300000
