#!/usr/bin/env bash
set -uo pipefail   # 不再用 `-e`，避免因出错提前退出

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$REPO_ROOT/preprocessing/log/json2vec"
LOCK_DIR="$REPO_ROOT/preprocessing/log/json2vec_lock"
mkdir -p "$LOG_DIR" "$LOCK_DIR"

completed=0

for id in $(seq 0 99); do
    lockfile="$LOCK_DIR/$id.lock"
    logfile="$LOG_DIR/$id.log"

    # 跳过已完成或正在执行的任务
    if [ -f "$lockfile" ] || [ -f "$logfile" ]; then
        continue
    fi

    # 原子创建锁文件
    if ( set -o noclobber; echo "$$" > "$lockfile" ) 2> /dev/null; then
        echo "==> Starting ID: $id"
        # 即使出错也不会终止整个脚本
        if /root/miniconda3/envs/pointercad/bin/python "$REPO_ROOT/preprocessing/json2vec.py" -c "$id" > "$logfile" 2>&1; then
            echo "==> Finished ID: $id"
        else
            echo "⚠️  ID $id 执行出错，日志保存在 $logfile"
        fi
        ((completed++))
    fi
done

# 统计信息
total=$(ls "$LOCK_DIR"/*.lock 2>/dev/null | wc -l || true)
echo "✅ 当前任务完成或锁定数: $total / 100"
echo "✅ 本次新增完成: $completed"
echo "全部处理完毕。"
