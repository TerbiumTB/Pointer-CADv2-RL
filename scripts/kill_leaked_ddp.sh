#!/usr/bin/env bash
# kill_leaked_torch_spawn.sh
#
# 安全清理因 torch.multiprocessing.spawn 崩溃而泄露的进程。
# 不依赖端口号，只检测进程环境变量中是否包含
#   WORLD_SIZE, MASTER_ADDR, MASTER_PORT, GLOBAL_RANK_OFFSET
# 等典型分布式训练变量。
#
# 使用示例：
#   ./scripts/kill_leaked_ddp.sh           # 交互确认后清理
#   ./scripts/kill_leaked_ddp.sh --yes     # 直接清理不询问
#
# 注意：
#   - 只清理当前用户的进程
#   - SIGTERM 等待 5 秒后 SIGKILL
#   - 需要 /proc 可访问（Linux）

set -euo pipefail

AUTO_YES=false
TERM_WAIT=5
ME_PID=$$
USER_ID=$(id -u)
ECHO_PREFIX="[kill_leaked_torch_spawn]"

info()  { echo "$ECHO_PREFIX $*"; }
warn()  { echo "$ECHO_PREFIX WARNING: $*" >&2; }
error() { echo "$ECHO_PREFIX ERROR: $*" >&2; }

# 解析参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    -y|--yes|--force) AUTO_YES=true; shift;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# //'
      exit 0
      ;;
    *)
      error "未知参数: $1"
      exit 1
      ;;
  esac
done

# 查找含分布式训练变量的进程（只看当前用户）
find_ddp_like_procs() {
  local -a pids=()
  for p in /proc/[0-9]*; do
    pid=$(basename "$p")
    if [[ "$pid" -eq "$ME_PID" ]]; then continue; fi
    # 只看当前用户的进程
    if [[ "$(stat -c %u "$p" 2>/dev/null || echo 0)" != "$USER_ID" ]]; then
      continue
    fi
    if [[ -r "$p/environ" ]]; then
      if tr '\0' '\n' < "$p/environ" 2>/dev/null | \
         grep -Eiq '(^|_)WORLD_SIZE=|(^|_)MASTER_ADDR=|(^|_)MASTER_PORT=|(^|_)GLOBAL_RANK_OFFSET='; then
         pids+=("$pid")
      fi
    fi
  done
  printf "%s\n" "${pids[@]}" | sort -u
}

perform_kill() {
  local -a pids=("$@")
  if [[ ${#pids[@]} -eq 0 ]]; then
    info "未发现疑似泄露进程。"
    return 0
  fi

  info "发现疑似 torch.spawn 泄露的进程:"
  for pid in "${pids[@]}"; do
    if [[ -r "/proc/$pid/cmdline" ]]; then
      cmd=$(tr '\0' ' ' < /proc/"$pid"/cmdline)
    else
      cmd=$(ps -p "$pid" -o args= 2>/dev/null || echo "[unknown]")
    fi
    echo "  PID=$pid CMD=$cmd"
  done

  if ! $AUTO_YES; then
    echo
    read -r -p "是否终止以上进程? [y/N] " ans
    case "$ans" in
      [Yy]|[Yy][Ee][Ss]) : ;;
      *) info "用户取消。"; exit 0;;
    esac
  fi

  info "发送 SIGTERM..."
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || warn "无法 SIGTERM $pid"
  done

  sleep "$TERM_WAIT"

  local -a remain=()
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then remain+=("$pid"); fi
  done

  if [[ ${#remain[@]} -gt 0 ]]; then
    warn "仍有 ${#remain[@]} 个进程存活，发送 SIGKILL..."
    for pid in "${remain[@]}"; do
      kill -9 "$pid" 2>/dev/null || warn "无法 SIGKILL $pid"
    done
  fi

  info "清理完成。"
}

main() {
  mapfile -t targets < <(find_ddp_like_procs)
  perform_kill "${targets[@]}"
}

main
