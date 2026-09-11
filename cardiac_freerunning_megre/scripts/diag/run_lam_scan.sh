#!/usr/bin/env bash
# Diagnostic only (scratchpad): wait for the 200-iteration check, then scan lambda on REF echo 0.
P=$(cat /data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/logs/run_v07_iter200_check.pid)
while kill -0 "$P" 2>/dev/null; do sleep 15; done
/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MEGRE/scripts/py /tmp/claude-1002/-home-congcongliu--claude/4016652d-b661-45ca-ba05-927ae4a7ee12/scratchpad/lam_scan.py
echo LAMSCAN_DONE
