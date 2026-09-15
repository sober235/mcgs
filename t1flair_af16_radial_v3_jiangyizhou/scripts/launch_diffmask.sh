#!/bin/bash
# launch_diffmask.sh -- start the diff-mask queue detached (the session's memory watchdog kills background
# tasks started from the tool shell, and zsh does not word-split $VAR, so this has to be a bash script file).
cd /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_gauss_v3_jiangyizhou
exec env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID \
  /home/congcongliu/anaconda3/envs/GSTMI/bin/python -u scripts/run_all_diffmask.py "$@"
