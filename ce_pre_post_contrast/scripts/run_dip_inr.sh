#!/bin/bash
W=/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast
S=$W/scripts
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
run() {  # gpu method scan extra...
  local gpu=$1 meth=$2 scan=$3; shift 3
  PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID \
    "$PY" -u "$S/ce_dip_inr.py" --method "$meth" --scan "$scan" --gpu "$gpu" --out "$W/dipinr" "$@" \
    > "$W/dipinr/logs/${meth}_${scan}.log" 2>&1
}
run 0 dip pre  --iters 5000 --lr 3e-3 --eval_every 100 &
run 1 dip post --iters 5000 --lr 3e-3 --eval_every 100 &
run 2 inr pre  --iters 1500 --eval_every 50 &
run 3 inr post --iters 1500 --eval_every 50 &
wait
echo "ALL DIP/INR DONE"
