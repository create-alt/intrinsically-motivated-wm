#!/bin/bash

cd /home/ist_baidoku/yoshinari.kawashima/wm25_final_homework/dreamerv3

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
echo "Starting multiple experiments"
echo "=========================================="

# GPU情報の確認
echo "GPU Status:"
nvidia-smi
echo ""

###############################################################################
# 実験1: seed=0
###############################################################################
EXP_NAME="exp1_seed0"
TIME_STR=$(date '+%y%m%d%H%M')
LOG_DIR="log/${TIME_STR}_dreamerV3_atari100k_ms_pacman_trendmix_${EXP_NAME}"
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "[Experiment 1] ${EXP_NAME}"
echo "Log directory: ${LOG_DIR}"
echo "Started at: $(date)"
echo "=========================================="

python dreamerv3/main.py \
    --configs atari100k \
    --task atari100k_ms_pacman \
    --run.train_ratio 128 \
    --logdir ${LOG_DIR} \
    --seed 0 \
    --agent.dormant.enable True \
    --agent.dormant.tau 0.025 \
    --replay.trend.enable True \
    --replay.trend.fast 0.1 \
    --replay.trend.slow 0.01 \
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
    2>&1 | tee ${LOG_DIR}/log.log

echo "[Experiment 1] Finished at: $(date)"
echo ""

###############################################################################
# 実験2: seed=1 (ここでハイパラを変更)
###############################################################################
EXP_NAME="exp2_seed1"
TIME_STR=$(date '+%y%m%d%H%M')
LOG_DIR="log/${TIME_STR}_dreamerV3_atari100k_ms_pacman_trendmix_${EXP_NAME}"
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "[Experiment 2] ${EXP_NAME}"
echo "Log directory: ${LOG_DIR}"
echo "Started at: $(date)"
echo "=========================================="

python dreamerv3/main.py \
    --configs atari100k \
    --task atari100k_ms_pacman \
    --run.train_ratio 128 \
    --logdir ${LOG_DIR} \
    --seed 1 \
    --agent.dormant.enable True \
    --agent.dormant.tau 0.025 \
    --replay.trend.enable True \
    --replay.trend.fast 0.1 \
    --replay.trend.slow 0.01 \
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
    2>&1 | tee ${LOG_DIR}/log.log

echo "[Experiment 2] Finished at: $(date)"
echo ""

###############################################################################
echo "=========================================="
echo "All experiments completed at: $(date)"
echo "=========================================="
