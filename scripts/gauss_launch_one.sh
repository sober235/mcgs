#!/usr/bin/env bash
# gauss_launch_one.sh EXP CARD ITERS [KEY=VAL ...] [-- extra args...]
# Runs one autoresearch_gauss.sh experiment on CARD while holding the shared card lock
# <cardiac topic>/work/.gpulock.<CARD> (mkdir-atomic, pid file = this wrapper), which the diff-mask and cardiac
# schedulers also honour; the lock is removed when the run ends, whatever the exit code.
set -uo pipefail
cd "$(dirname "$0")/.."
EXP=$1; CARD=$2; IT=$3; shift 3
ENVS=()
while [[ $# -gt 0 && "$1" == *=* && "$1" != -* ]]; do ENVS+=("$1"); shift; done
[[ "${1:-}" == "--" ]] && shift
LOCK=/data0/congcong/code/GS/Results_20260908/Cardiac_FreeRunning_MCGS/work/.gpulock.$CARD
if ! mkdir "$LOCK" 2>/dev/null; then
  owner=$(cat "$LOCK/pid" 2>/dev/null)
  if [[ -n "$owner" ]] && kill -0 "$owner" 2>/dev/null; then echo "card $CARD locked by live pid $owner" >&2; exit 3; fi
  rm -rf "$LOCK"; mkdir "$LOCK" || exit 3
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT
env "${ENVS[@]}" ./autoresearch_gauss.sh "$EXP" "$CARD" "$IT" "$@"
