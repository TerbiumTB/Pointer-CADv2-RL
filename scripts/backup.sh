#!/usr/bin/env bash
# watch_model_backup.sh
# 每 10 分钟扫描一次 ./log 下的所有子目录，找到大于 800MiB 的 model.pth 并就地备份为 model.pth.bk

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$REPO_ROOT/log"
INTERVAL=600   # 10 分钟 = 600 秒

scan_and_backup() {
  [[ -d "$LOG_DIR" ]] || return 0

  while IFS= read -r -d '' f; do
    bk="${f}.bk"
    if [[ ! -e "$bk" || "$f" -nt "$bk" ]]; then
      echo "[ACTION] 备份: $f -> $bk"
      cp --reflink=auto --sparse=always -f "$f" "$bk"
      touch -r "$f" "$bk"
      echo "[OK] 已备份: $bk"
    fi
  done < <(find "$LOG_DIR" -type f -name 'model.pth' -size +800M -print0)
}

trap 'echo; echo "[EXIT] 用户中断，退出监控。"; exit 0' INT TERM

echo "[INFO] 每 ${INTERVAL}s (10分钟) 扫描一次 $LOG_DIR"

while true; do
  scan_and_backup
  sleep "$INTERVAL"
done
