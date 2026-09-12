#!/bin/bash
# p2_seed43_of_best.sh CARD: wait until the mechanical grid promotion (p2_best_rot<rl>_slr<sl>_full4k) has been launched,
# then run the same recipe with seed 43 at 4000 iterations with volumes on CARD (the seed repeat = the floor of the
# promoted recipe and one half of its two-seed row). Values are parsed from the promotion's name.
cd /data0/congcong/code/GS/mcgs_af12_wt
G=${1:?card}
while :; do f=$(ls results_arc/p2_best_rot*_slr*_full4k.log 2>/dev/null | head -1); [ -n "$f" ] && break; sleep 60; done
name=$(basename "$f" .log)
rl=$(echo "$name" | sed -E 's/^p2_best_rot([0-9]+)_slr([0-9]+)_full4k$/0.\1/')
sl=$(echo "$name" | sed -E 's/^p2_best_rot([0-9]+)_slr([0-9]+)_full4k$/0.\2/')
echo "[seed43_of_best] promotion $name -> rotation_lr=$rl scale_lr=$sl; launching ${name%_full4k}_s43_full4k on card $G"
SAVE_VOLUMES=1 ./autoresearch_jyz_af12_p2g.sh "${name%_full4k}_s43_full4k" "$G" 4000 --set gaussian.rotation_lr="$rl" --set gaussian.scale_lr="$sl" --set training.seed=43
