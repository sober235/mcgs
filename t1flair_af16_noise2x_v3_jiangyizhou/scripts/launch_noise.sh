#!/bin/bash
# launch_noise.sh [run_all_noise.py args] -- start the noise-study queue detached (the tool shell is zsh and its
# background tasks are killed by the session's memory watchdog, so this has to be a bash script file).
cd /data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_noise2x_v3_jiangyizhou || exit 1
exec env PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID \
  /home/congcongliu/anaconda3/envs/GSTMI/bin/python -u scripts/run_all_noise.py "$@"
