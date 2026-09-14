# card_ok_af24.sh -- sourced by the AF24 launchers. card_ok GPU NEED_GB is true when none of MY compute processes is on
# that card and the card has NEED_GB free. Every card carried another user's long job when the AF24 study started
# (2026-09-14 21:00), so the AF20 rule "start only on a card with no compute process at all" would never fire; one
# trainer of mine per card is the rule that stays (two MC-GS trainers on one card hung it twice on 2026-09-11).
# nvidia-smi output goes into variables before any grep: under pipefail `nvidia-smi | grep -q` fails when grep exits
# early (SIGPIPE) and reported busy cards as idle on 2026-09-14.
card_ok() {
  local g=$1 need=$2 me bus free apps pid b
  me=$(id -u)
  bus=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader) || return 1
  bus=$(awk -F', ' -v i="$g" '$1==i{print $2}' <<< "$bus")
  [ -n "$bus" ] || return 1
  free=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits) || return 1
  free=$(awk -F', ' -v i="$g" '$1==i{print $2}' <<< "$free")
  [ -n "$free" ] && [ "$free" -ge $((need * 1024)) ] || return 1
  apps=$(nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader) || return 1
  while IFS=', ' read -r pid b; do
    [ -n "$pid" ] && [ "$b" = "$bus" ] || continue
    [ "$(stat -c %u /proc/$pid 2>/dev/null)" = "$me" ] && return 1
  done <<< "$apps"
  return 0
}
