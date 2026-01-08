#!/bin/bash

# セルフデタッチ: 引数なしで呼ばれたら、自分自身をnohupでバックグラウンド実行
if [ "$1" != "--running" ]; then
    MASTER_LOG="log/multi_experiment_$(date '+%y%m%d%H%M').log"
    mkdir -p log
    nohup "$0" --running > "$MASTER_LOG" 2>&1 &
    PID=$!
    echo "=========================================="
    echo "All 6 experiments started in background"
    echo "  PID: $PID"
    echo "  Master log: $MASTER_LOG"
    echo "=========================================="
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
echo "Starting multiple experiments"
echo "Started at: $(date)"
echo "=========================================="

# GPU情報の確認
echo "GPU Status:"
nvidia-smi
echo ""

###############################################################################
# 実験1: uniform=0.5
###############################################################################
EXP_NAME="exp01_uniform05"
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
    --replay.fracs.uniform 0.5 \
    --replay.fracs.priority 0.0 \
    --replay.fracs.recency 0.0 \
    --replay.fracs.curious 0.0 \
    --replay.fracs.explore 0.25 \
    --replay.fracs.exploit 0.25 \
    --jax.platform cuda \
    --logger.outputs jsonl,wandb \
    2>&1 | tee ${LOG_DIR}/log.log

echo "[Experiment 1] Finished at: $(date)"
echo ""

###############################################################################
# 実験2: uniform=0.3
###############################################################################
EXP_NAME="exp02_uniform03"
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
    --replay.fracs.uniform 0.3 \
    --replay.fracs.priority 0.0 \
    --replay.fracs.recency 0.0 \
    --replay.fracs.curious 0.0 \
    --replay.fracs.explore 0.35 \
    --replay.fracs.exploit 0.35 \
    --jax.platform cuda \
    --logger.outputs jsonl,wandb \
    2>&1 | tee ${LOG_DIR}/log.log

echo "[Experiment 2] Finished at: $(date)"
echo ""

###############################################################################
# 実験3: fast=0.3, slow=0.03
###############################################################################
EXP_NAME="exp03_fast03_slow003"
TIME_STR=$(date '+%y%m%d%H%M')
LOG_DIR="log/${TIME_STR}_dreamerV3_atari100k_ms_pacman_trendmix_${EXP_NAME}"
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "[Experiment 3] ${EXP_NAME}"
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
    --replay.trend.fast 0.3 \
    --replay.trend.slow 0.03 \
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

echo "[Experiment 3] Finished at: $(date)"
echo ""


###############################################################################
# 実験4: fast=0.05, slow=0.005
###############################################################################
EXP_NAME="exp04_fast005_slow0005"
TIME_STR=$(date '+%y%m%d%H%M')
LOG_DIR="log/${TIME_STR}_dreamerV3_atari100k_ms_pacman_trendmix_${EXP_NAME}"
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "[Experiment 4] ${EXP_NAME}"
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
    --replay.trend.fast 0.05 \
    --replay.trend.slow 0.005 \
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

echo "[Experiment 4] Finished at: $(date)"
echo ""


###############################################################################
# 実験5: clip=0.3
###############################################################################
EXP_NAME="exp05_clip03"
TIME_STR=$(date '+%y%m%d%H%M')
LOG_DIR="log/${TIME_STR}_dreamerV3_atari100k_ms_pacman_trendmix_${EXP_NAME}"
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "[Experiment 5] ${EXP_NAME}"
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
    --replay.trend.gate_min 0.3 \
    --replay.trend.gate_max 0.7 \
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

echo "[Experiment 5] Finished at: $(date)"
echo ""


###############################################################################
# 実験6: k=0.01
###############################################################################
EXP_NAME="exp06_k001"
TIME_STR=$(date '+%y%m%d%H%M')
LOG_DIR="log/${TIME_STR}_dreamerV3_atari100k_ms_pacman_trendmix_${EXP_NAME}"
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "[Experiment 6] ${EXP_NAME}"
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
    --replay.trend.k 0.01 \
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

echo "[Experiment 6] Finished at: $(date)"
echo ""

###############################################################################
echo "=========================================="
echo "All experiments completed at: $(date)"
echo "=========================================="
