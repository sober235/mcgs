#!/bin/bash
# run_classical_revisited_ext.sh -- extends the TV sweep DOWNWARD: on this subject at AF16 the converged-TV
# optimum sat on the lower edge of the first grid (T1 0.0007 > 0.001; FLAIR 0.0005 > 0.0007), so the
# upward points (T1 0.002, FLAIR 0.0014) were dropped and two lower points added per contrast; the joint
# vectorial TV (a weaker penalty per contrast at the same lambda) is swept at {0.7, 1}e-3.
J=/data0/congcong/code/GS/Results_20260908/T1FLAIR_AF16_v3_jiangyizhou
PY=/home/congcongliu/anaconda3/envs/GSTMI/bin/python
export PYTHONNOUSERSITE=1 PYTHONPATH= CUDA_DEVICE_ORDER=PCI_BUS_ID
cd $J
tv()  { local g=$1 c=$2 L=$3; [ -f logs/classical_${c}_o200_lam${L}.json ] && return
        $PY -u scripts/recon_classical.py --contrast $c --gpu $g --methods cssense --admm_outer 200 --lam $L --suffix _o200_lam$L > logs/classical_${c}_o200_lam${L}.log 2>&1
        echo "[$(date +%H:%M)] tv $c lam=$L rc=$?" >> logs/classical_revisited.log; }
jtv() { local g=$1 L=$2; [ -f logs/jcssense_o200_lam${L}.json ] && return
        $PY -u scripts/recon_joint_cssense.py --topic $J --contrasts T1,FLAIR --gpu $g --admm_outer 200 --lam $L --suffix _o200_lam$L > logs/jcssense_o200_lam${L}.log 2>&1
        echo "[$(date +%H:%M)] jtv lam=$L rc=$?" >> logs/classical_revisited.log; }
( tv 2 T1 0.0005; tv 2 T1 0.00035; jtv 2 0.0007; jtv 2 0.001; echo "[$(date +%H:%M)] CHAIN_A2_DONE" >> logs/classical_revisited.log ) &
( tv 3 FLAIR 0.00035; tv 3 FLAIR 0.00025; echo "[$(date +%H:%M)] CHAIN_B2_DONE" >> logs/classical_revisited.log ) &
wait
