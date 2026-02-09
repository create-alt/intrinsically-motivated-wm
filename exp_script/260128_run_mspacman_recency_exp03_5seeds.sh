#!/bin/bash

# セルフデタッチ: 引数なしで呼ばれたら、自分自身をnohupでバックグラウンド実行
if [ "$1" != "--running" ]; then
    MASTER_LOG="log/mspacman_recency_exp03_$(date '+%y%m%d%H%M').log"
    mkdir -p log
    nohup "$0" --running > "$MASTER_LOG" 2>&1 &
    PID=$!
    echo "=========================================="
    echo "Ms Pacman: Recency Sampling with recexp=0.3 (5 seeds)"
    echo "  PID: $PID"
    echo "  Master log: $MASTER_LOG"
    echo "=========================================="
    echo ""
    echo "Experiments (5 total):"
    echo "  Recency sampling (recexp=0.3): seed 0-4"
    echo ""
    echo "Common settings:"
    echo "  - agent.dormant.enable: True"
    echo "  - agent.dormant.tau: 0.025"
    echo "  - logger.videos: False"
    echo ""
    echo "Recency settings:"
    echo "  - replay.fracs.recency: 1.0"
    echo "  - replay.fracs.uniform: 0.0"
    echo "  - replay.recexp: 0.3 (moderate decay)"
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
echo "Starting Ms Pacman: Recency Sampling (recexp=0.3) experiments"
echo "Started at: $(date)"
echo "=========================================="

# GPU情報の確認
echo "GPU Status:"
nvidia-smi
echo ""

TASK="ms_pacman"
EXP_COUNT=0

###############################################################################
# Recency実験 (recexp=0.3) - 5 seeds
###############################################################################
for SEED in 0 1 2 3 4; do
    EXP_COUNT=$((EXP_COUNT + 1))
    EXP_NAME="recency_exp03_seed${SEED}"
    TIME_STR=$(date '+%y%m%d%H%M')
    LOG_DIR="log/${TIME_STR}_dreamerV3_atari100k_${TASK}_${EXP_NAME}"
    mkdir -p ${LOG_DIR}

    echo "=========================================="
    echo "[Experiment ${EXP_COUNT}/5] ${TASK} - ${EXP_NAME}"
    echo "Log directory: ${LOG_DIR}"
    echo "Started at: $(date)"
    echo "=========================================="

    python dreamerv3/main.py \
        --configs atari100k \
        --task atari100k_${TASK} \
        --run.train_ratio 128 \
        --logdir ${LOG_DIR} \
        --seed ${SEED} \
        --agent.dormant.enable True \
        --agent.dormant.tau 0.025 \
        --replay.fracs.uniform 0.0 \
        --replay.fracs.recency 1.0 \
        --replay.recexp 0.3 \
        --jax.platform cuda \
        --logger.outputs jsonl,wandb \
        --logger.videos False \
        2>&1 | tee ${LOG_DIR}/log.log

    echo "[Experiment ${EXP_COUNT}/5] ${TASK} - ${EXP_NAME} Finished at: $(date)"
    echo ""
done

###############################################################################
echo "=========================================="
echo "All experiments completed at: $(date)"
echo "Total: 5 experiments (recency sampling, recexp=0.3)"
echo "=========================================="
