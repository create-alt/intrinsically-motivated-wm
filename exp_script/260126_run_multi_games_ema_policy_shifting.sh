#!/bin/bash

# セルフデタッチ: 引数なしで呼ばれたら、自分自身をnohupでバックグラウンド実行
if [ "$1" != "--running" ]; then
    MASTER_LOG="log/multi_experiment_ema_policy_shifting_$(date '+%y%m%d%H%M').log"
    mkdir -p log
    nohup "$0" --running > "$MASTER_LOG" 2>&1 &
    PID=$!
    echo "=========================================="
    echo "All 5 EMA-Based Policy Shifting experiments started in background"
    echo "  PID: $PID"
    echo "  Master log: $MASTER_LOG"
    echo "=========================================="
    echo ""
    echo "Experiments (EMA-Based Policy Shifting enabled):"
    echo "  1. Bank Heist - ema_policy_shifting"
    echo "  2. Frostbite - ema_policy_shifting"
    echo "  3. Hero - ema_policy_shifting"
    echo "  4. Seaquest - ema_policy_shifting"
    echo "  5. Ms Pacman - ema_policy_shifting"
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
echo "Starting EMA-Based Policy Shifting experiments"
echo "Started at: $(date)"
echo "=========================================="

# GPU情報の確認
echo "GPU Status:"
nvidia-smi
echo ""

###############################################################################
# 実験1: Bank Heist - ema_policy_shifting
###############################################################################
TASK="bank_heist"
EXP_NAME="ema_policy_shifting"
TIME_STR=$(date '+%y%m%d%H%M')
LOG_DIR="log/${TIME_STR}_dreamerV3_atari100k_${TASK}_${EXP_NAME}"
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "[Experiment 1] ${TASK} - ${EXP_NAME}"
echo "Log directory: ${LOG_DIR}"
echo "Started at: $(date)"
echo "=========================================="

python dreamerv3/main.py \
    --configs atari100k \
    --task atari100k_${TASK} \
    --run.train_ratio 128 \
    --logdir ${LOG_DIR} \
    --seed 0 \
    --agent.dormant.enable True \
    --agent.dormant.tau 0.025 \
    --agent.ema_policy_shifting.enable True \
    --jax.platform cuda \
    --jax.profiler False \
    --logger.outputs jsonl,wandb \
    --logger.videos False \
    2>&1 | tee ${LOG_DIR}/log.log

echo "[Experiment 1] ${TASK} - ${EXP_NAME} Finished at: $(date)"
echo ""

###############################################################################
# 実験2: Frostbite - ema_policy_shifting
###############################################################################
TASK="frostbite"
EXP_NAME="ema_policy_shifting"
TIME_STR=$(date '+%y%m%d%H%M')
LOG_DIR="log/${TIME_STR}_dreamerV3_atari100k_${TASK}_${EXP_NAME}"
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "[Experiment 2] ${TASK} - ${EXP_NAME}"
echo "Log directory: ${LOG_DIR}"
echo "Started at: $(date)"
echo "=========================================="

python dreamerv3/main.py \
    --configs atari100k \
    --task atari100k_${TASK} \
    --run.train_ratio 128 \
    --logdir ${LOG_DIR} \
    --seed 0 \
    --agent.dormant.enable True \
    --agent.dormant.tau 0.025 \
    --agent.ema_policy_shifting.enable True \
    --jax.platform cuda \
    --jax.profiler False \
    --logger.outputs jsonl,wandb \
    --logger.videos False \
    2>&1 | tee ${LOG_DIR}/log.log

echo "[Experiment 2] ${TASK} - ${EXP_NAME} Finished at: $(date)"
echo ""

###############################################################################
# 実験3: Hero - ema_policy_shifting
###############################################################################
TASK="hero"
EXP_NAME="ema_policy_shifting"
TIME_STR=$(date '+%y%m%d%H%M')
LOG_DIR="log/${TIME_STR}_dreamerV3_atari100k_${TASK}_${EXP_NAME}"
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "[Experiment 3] ${TASK} - ${EXP_NAME}"
echo "Log directory: ${LOG_DIR}"
echo "Started at: $(date)"
echo "=========================================="

python dreamerv3/main.py \
    --configs atari100k \
    --task atari100k_${TASK} \
    --run.train_ratio 128 \
    --logdir ${LOG_DIR} \
    --seed 0 \
    --agent.dormant.enable True \
    --agent.dormant.tau 0.025 \
    --agent.ema_policy_shifting.enable True \
    --jax.platform cuda \
    --jax.profiler False \
    --logger.outputs jsonl,wandb \
    --logger.videos False \
    2>&1 | tee ${LOG_DIR}/log.log

echo "[Experiment 3] ${TASK} - ${EXP_NAME} Finished at: $(date)"
echo ""

###############################################################################
# 実験4: Seaquest - ema_policy_shifting
###############################################################################
TASK="seaquest"
EXP_NAME="ema_policy_shifting"
TIME_STR=$(date '+%y%m%d%H%M')
LOG_DIR="log/${TIME_STR}_dreamerV3_atari100k_${TASK}_${EXP_NAME}"
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "[Experiment 4] ${TASK} - ${EXP_NAME}"
echo "Log directory: ${LOG_DIR}"
echo "Started at: $(date)"
echo "=========================================="

python dreamerv3/main.py \
    --configs atari100k \
    --task atari100k_${TASK} \
    --run.train_ratio 128 \
    --logdir ${LOG_DIR} \
    --seed 0 \
    --agent.dormant.enable True \
    --agent.dormant.tau 0.025 \
    --agent.ema_policy_shifting.enable True \
    --jax.platform cuda \
    --jax.profiler False \
    --logger.outputs jsonl,wandb \
    --logger.videos False \
    2>&1 | tee ${LOG_DIR}/log.log

echo "[Experiment 4] ${TASK} - ${EXP_NAME} Finished at: $(date)"
echo ""

###############################################################################
# 実験5: Ms Pacman - ema_policy_shifting
###############################################################################
TASK="ms_pacman"
EXP_NAME="ema_policy_shifting"
TIME_STR=$(date '+%y%m%d%H%M')
LOG_DIR="log/${TIME_STR}_dreamerV3_atari100k_${TASK}_${EXP_NAME}"
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "[Experiment 5] ${TASK} - ${EXP_NAME}"
echo "Log directory: ${LOG_DIR}"
echo "Started at: $(date)"
echo "=========================================="

python dreamerv3/main.py \
    --configs atari100k \
    --task atari100k_${TASK} \
    --run.train_ratio 128 \
    --logdir ${LOG_DIR} \
    --seed 0 \
    --agent.dormant.enable True \
    --agent.dormant.tau 0.025 \
    --agent.ema_policy_shifting.enable True \
    --jax.platform cuda \
    --jax.profiler False \
    --logger.outputs jsonl,wandb \
    --logger.videos False \
    2>&1 | tee ${LOG_DIR}/log.log

echo "[Experiment 5] ${TASK} - ${EXP_NAME} Finished at: $(date)"
echo ""

###############################################################################
echo "=========================================="
echo "All EMA-Based Policy Shifting experiments completed at: $(date)"
echo "=========================================="
