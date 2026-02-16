#!/bin/bash

# セルフデタッチ: 引数なしで呼ばれたら、自分自身をnohupでバックグラウンド実行
if [ "$1" != "--running" ]; then
    MASTER_LOG="log/freeway_ema_policy_shifting_seed234_$(date '+%y%m%d%H%M').log"
    mkdir -p log
    nohup "$0" --running > "$MASTER_LOG" 2>&1 &
    PID=$!
    echo "=========================================="
    echo "Freeway: EMA-Based Policy Shifting (seed 2,3,4)"
    echo "  PID: $PID"
    echo "  Master log: $MASTER_LOG"
    echo "=========================================="
    echo ""
    echo "Experiments (3 total):"
    echo "  EMA-Based Policy Shifting (dormant + ema_policy_shifting + dist:normal): seed 2,3,4"
    echo ""
    echo "Common settings:"
    echo "  - agent.dyn.rssm.dist: normal"
    echo "  - agent.ema_policy_shifting.enable: True"
    echo "  - agent.ema_policy_shifting.ema_span_short: 10"
    echo "  - agent.ema_policy_shifting.ema_span_long: 1000"
    echo "  - agent.ema_policy_shifting.target_std_small: 0.1"
    echo "  - agent.ema_policy_shifting.target_std_large: 1.0"
    echo "  - agent.ema_policy_shifting.explore_limit: 5"
    echo "  - agent.ema_policy_shifting.default_force: 2"
    echo "  - agent.dormant.enable: True"
    echo "  - agent.dormant.tau: 0.025"
    echo "  - logger.videos: False"
    echo ""
    echo "Useful commands:"
    echo "  Monitor:       tail -f $MASTER_LOG"
    echo "  Check process: ps aux | grep $PID"
    echo "  Stop all:      kill $PID"
    echo "=========================================="
    exit 0
fi

# 以下はバックグラウンドで実行される部分
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# 環境変数を読み込む
if [ -f .env ]; then
    echo "Loading environment variables from .env"
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Warning: .env file not found"
fi

# 仮想環境をアクティベート
source .venv/bin/activate

echo "=========================================="
echo "Starting Freeway: EMA-Based Policy Shifting experiments (seed 2,3,4)"
echo "Started at: $(date)"
echo "=========================================="

# GPU情報の確認
echo "GPU Status:"
nvidia-smi
echo ""

TASK="freeway"
EXP_COUNT=0

###############################################################################
# EMA-Based Policy Shifting 実験 (dormant + ema_policy_shifting + dist:normal) - seed 2,3,4
###############################################################################
for SEED in 2 3 4; do
    EXP_COUNT=$((EXP_COUNT + 1))
    EXP_NAME="ema_policy_shifting_seed${SEED}"
    TIME_STR=$(date '+%y%m%d%H%M')
    LOG_DIR="log/${TIME_STR}_dreamerV3_atari100k_${TASK}_${EXP_NAME}"
    mkdir -p ${LOG_DIR}

    echo "=========================================="
    echo "[Experiment ${EXP_COUNT}/3] ${TASK} - ${EXP_NAME}"
    echo "Log directory: ${LOG_DIR}"
    echo "Started at: $(date)"
    echo "=========================================="

    python dreamerv3/main.py \
        --configs atari100k \
        --task atari100k_${TASK} \
        --run.train_ratio 128 \
        --logdir ${LOG_DIR} \
        --seed ${SEED} \
        --agent.dyn.rssm.dist normal \
        --agent.ema_policy_shifting.enable True \
        --agent.ema_policy_shifting.ema_span_short 10 \
        --agent.ema_policy_shifting.ema_span_long 1000 \
        --agent.ema_policy_shifting.target_std_small 0.1 \
        --agent.ema_policy_shifting.target_std_large 1.0 \
        --agent.ema_policy_shifting.explore_limit 5 \
        --agent.ema_policy_shifting.default_force 2 \
        --agent.dormant.enable True \
        --agent.dormant.tau 0.025 \
        --jax.platform cuda \
        --jax.profiler False \
        --logger.outputs jsonl,wandb \
        --logger.videos False \
        2>&1 | tee ${LOG_DIR}/log.log

    echo "[Experiment ${EXP_COUNT}/3] ${TASK} - ${EXP_NAME} Finished at: $(date)"
    echo ""
done

###############################################################################
echo "=========================================="
echo "All experiments completed at: $(date)"
echo "Total: 3 experiments (seed 2,3,4)"
echo "=========================================="
