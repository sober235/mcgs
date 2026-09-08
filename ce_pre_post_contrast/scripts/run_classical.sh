#!/bin/bash
W=/data0/congcong/code/GS/Results_20260908/CE_pre_post_contrast
S=$W/scripts
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
run() {  # gpu method scan [extra...]
  local gpu=$1 meth=$2 scan=$3; shift 3
  PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID \
    "$PY" -u "$S/ce_classical.py" --method "$meth" --scan "$scan" --gpu "$gpu" --out "$W/classical" "$@" \
    > "$W/classical/logs/${meth}_${scan}.log" 2>&1
}
( run 3 zf post; run 3 cgsense post; run 3 cssense post ) &
( run 5 zsssdu pre  --epochs 100 ) &
( run 6 zsssdu post --epochs 100 ) &
wait
echo "ALL CLASSICAL DONE"
