#!/bin/bash

# セルフデタッチ: 引数なしで呼ばれたら、自分自身をnohupでバックグラウンド実行
if [ "$1" != "--running" ]; then
    MASTER_LOG="log/trendmix_exp09_5games_curious_hybrid_$(date '+%y%m%d%H%M').log"
    mkdir -p log
    nohup "$0" --running > "$MASTER_LOG" 2>&1 &
    PID=$!
    echo "=========================================="
    echo "TrendMixture exp09 (curious + hybrid): 5 experiments started in background"
    echo "  PID: $PID"
    echo "  Master log: $MASTER_LOG"
    echo "=========================================="
    echo ""
    echo "Experiments:"
    echo "  1. Bank Heist - trendmix_exp09_curious_hybrid"
    echo "  2. Frostbite - trendmix_exp09_curious_hybrid"
    echo "  3. Hero - trendmix_exp09_curious_hybrid"
    echo "  4. Seaquest - trendmix_exp09_curious_hybrid"
    echo "  5. Ms Pacman - trendmix_exp09_curious_hybrid"
    echo ""
    echo "exp09 settings (curious replay + velocity-accel hybrid):"
    echo "  - replay.trend.enable: True"
    echo "  - replay.trend.priority_mode: curious"
    echo "  - replay.trend.velocity_frac: 0.5 (50% velocity + 50% accel)"
    echo "  - replay.trend.curious.c: 1e4"
    echo "  - replay.trend.curious.beta: 0.7"
    echo "  - replay.trend.fast: 0.01, slow: 0.001"
    echo "  - replay.trend.k: 5.0"
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
echo "Starting TrendMixture exp09 experiments (5 games, curious + hybrid)"
echo "Started at: $(date)"
echo "=========================================="

# GPU情報の確認
echo "GPU Status:"
nvidia-smi
echo ""

###############################################################################
# 実験1: Bank Heist - trendmix_exp09_curious_hybrid
###############################################################################
TASK="bank_heist"
EXP_NAME="trendmix_exp09_curious_hybrid"
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
    --replay.trend.enable True \
    --replay.trend.priority_mode curious \
    --replay.trend.velocity_frac 0.5 \
    --replay.trend.curious.c 1e4 \
    --replay.trend.curious.beta 0.7 \
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

echo "[Experiment 1] ${TASK} - ${EXP_NAME} Finished at: $(date)"
echo ""

###############################################################################
# 実験2: Frostbite - trendmix_exp09_curious_hybrid
###############################################################################
TASK="frostbite"
EXP_NAME="trendmix_exp09_curious_hybrid"
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
    --replay.trend.enable True \
    --replay.trend.priority_mode curious \
    --replay.trend.velocity_frac 0.5 \
    --replay.trend.curious.c 1e4 \
    --replay.trend.curious.beta 0.7 \
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

echo "[Experiment 2] ${TASK} - ${EXP_NAME} Finished at: $(date)"
echo ""

###############################################################################
# 実験3: Hero - trendmix_exp09_curious_hybrid
###############################################################################
TASK="hero"
EXP_NAME="trendmix_exp09_curious_hybrid"
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
    --replay.trend.enable True \
    --replay.trend.priority_mode curious \
    --replay.trend.velocity_frac 0.5 \
    --replay.trend.curious.c 1e4 \
    --replay.trend.curious.beta 0.7 \
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

echo "[Experiment 3] ${TASK} - ${EXP_NAME} Finished at: $(date)"
echo ""

###############################################################################
# 実験4: Seaquest - trendmix_exp09_curious_hybrid
###############################################################################
TASK="seaquest"
EXP_NAME="trendmix_exp09_curious_hybrid"
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
    --replay.trend.enable True \
    --replay.trend.priority_mode curious \
    --replay.trend.velocity_frac 0.5 \
    --replay.trend.curious.c 1e4 \
    --replay.trend.curious.beta 0.7 \
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

echo "[Experiment 4] ${TASK} - ${EXP_NAME} Finished at: $(date)"
echo ""

###############################################################################
# 実験5: Ms Pacman - trendmix_exp09_curious_hybrid
###############################################################################
TASK="ms_pacman"
EXP_NAME="trendmix_exp09_curious_hybrid"
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
    --replay.trend.enable True \
    --replay.trend.priority_mode curious \
    --replay.trend.velocity_frac 0.5 \
    --replay.trend.curious.c 1e4 \
    --replay.trend.curious.beta 0.7 \
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

echo "[Experiment 5] ${TASK} - ${EXP_NAME} Finished at: $(date)"
echo ""

###############################################################################
echo "=========================================="
echo "All TrendMixture exp09 (curious + hybrid) experiments completed at: $(date)"
echo "=========================================="
