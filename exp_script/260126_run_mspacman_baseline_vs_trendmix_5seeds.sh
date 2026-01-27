#!/bin/bash

# セルフデタッチ: 引数なしで呼ばれたら、自分自身をnohupでバックグラウンド実行
if [ "$1" != "--running" ]; then
    MASTER_LOG="log/mspacman_baseline_vs_trendmix_$(date '+%y%m%d%H%M').log"
    mkdir -p log
    nohup "$0" --running > "$MASTER_LOG" 2>&1 &
    PID=$!
    echo "=========================================="
    echo "Ms Pacman: Baseline vs TrendMix (5 seeds each)"
    echo "  PID: $PID"
    echo "  Master log: $MASTER_LOG"
    echo "=========================================="
    echo ""
    echo "Experiments (10 total):"
    echo "  Baseline (dormant only): seed 0-4"
    echo "  TrendMix: seed 0-4"
    echo ""
    echo "Common settings:"
    echo "  - agent.dormant.enable: True"
    echo "  - agent.dormant.tau: 0.025"
    echo "  - logger.videos: False"
    echo ""
    echo "TrendMix additional settings:"
    echo "  - replay.trend.enable: True"
    echo "  - replay.trend.eps: 1e-6"
    echo "  - replay.fracs: explore=0.5, exploit=0.5"
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
echo "Starting Ms Pacman: Baseline vs TrendMix experiments"
echo "Started at: $(date)"
echo "=========================================="

# GPU情報の確認
echo "GPU Status:"
nvidia-smi
echo ""

TASK="ms_pacman"
EXP_COUNT=0

###############################################################################
# ベースライン実験 (dormant only) - 5 seeds
###############################################################################
for SEED in 0 1 2 3 4; do
    EXP_COUNT=$((EXP_COUNT + 1))
    EXP_NAME="baseline_dormant_seed${SEED}"
    TIME_STR=$(date '+%y%m%d%H%M')
    LOG_DIR="log/${TIME_STR}_dreamerV3_atari100k_${TASK}_${EXP_NAME}"
    mkdir -p ${LOG_DIR}

    echo "=========================================="
    echo "[Experiment ${EXP_COUNT}/10] ${TASK} - ${EXP_NAME}"
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
        --jax.platform cuda \
        --logger.outputs jsonl,wandb \
        --logger.videos False \
        2>&1 | tee ${LOG_DIR}/log.log

    echo "[Experiment ${EXP_COUNT}/10] ${TASK} - ${EXP_NAME} Finished at: $(date)"
    echo ""
done

###############################################################################
# TrendMix実験 - 5 seeds
###############################################################################
for SEED in 0 1 2 3 4; do
    EXP_COUNT=$((EXP_COUNT + 1))
    EXP_NAME="trendmix_seed${SEED}"
    TIME_STR=$(date '+%y%m%d%H%M')
    LOG_DIR="log/${TIME_STR}_dreamerV3_atari100k_${TASK}_${EXP_NAME}"
    mkdir -p ${LOG_DIR}

    echo "=========================================="
    echo "[Experiment ${EXP_COUNT}/10] ${TASK} - ${EXP_NAME}"
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
        --replay.trend.enable True \
        --replay.trend.fast 0.01 \
        --replay.trend.slow 0.001 \
        --replay.trend.k 5.0 \
        --replay.trend.eps 1e-6 \
        --replay.trend.gate_min 0.05 \
        --replay.trend.gate_max 0.95 \
        --replay.trend.gate_init 0.5 \
        --replay.fracs.uniform 0.0 \
        --replay.fracs.priority 0.0 \
        --replay.fracs.recency 0.0 \
        --replay.fracs.curious 0.0 \
        --replay.fracs.explore 0.5 \
        --replay.fracs.exploit 0.5 \
        --jax.platform cuda \
        --logger.outputs jsonl,wandb \
        --logger.videos False \
        2>&1 | tee ${LOG_DIR}/log.log

    echo "[Experiment ${EXP_COUNT}/10] ${TASK} - ${EXP_NAME} Finished at: $(date)"
    echo ""
done

###############################################################################
echo "=========================================="
echo "All experiments completed at: $(date)"
echo "Total: 10 experiments (5 baseline + 5 trendmix)"
echo "=========================================="
