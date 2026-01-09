#!/bin/bash

# セルフデタッチ: 引数なしで呼ばれたら、自分自身をnohupでバックグラウンド実行
if [ "$1" != "--running" ]; then
    MASTER_LOG="log/multi_experiment_normal_comparison_$(date '+%y%m%d%H%M').log"
    mkdir -p log
    nohup "$0" --running > "$MASTER_LOG" 2>&1 &
    PID=$!
    echo "=========================================="
    echo "All 12 experiments started in background"
    echo "  PID: $PID"
    echo "  Master log: $MASTER_LOG"
    echo "=========================================="
    echo ""
    echo "Experiments (baseline and normal alternating):"
    echo "  1. Bank Heist - baseline"
    echo "  2. Bank Heist - normal"
    echo "  3. Frostbite - baseline"
    echo "  4. Frostbite - normal"
    echo "  5. Hero - baseline"
    echo "  6. Hero - normal"
    echo "  7. Kangaroo - baseline"
    echo "  8. Kangaroo - normal"
    echo "  9. Alien - baseline"
    echo " 10. Alien - normal"
    echo " 11. Private Eye - baseline"
    echo " 12. Private Eye - normal"
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
echo "Starting multiple experiments (normal comparison)"
echo "Started at: $(date)"
echo "=========================================="

# GPU情報の確認
echo "GPU Status:"
nvidia-smi
echo ""

###############################################################################
# 実験1: Bank Heist - baseline
###############################################################################
TASK="bank_heist"
EXP_NAME="baseline"
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
    --jax.platform cuda \
    --logger.outputs jsonl,wandb \
    --logger.videos False \
    2>&1 | tee ${LOG_DIR}/log.log

echo "[Experiment 1] ${TASK} - ${EXP_NAME} Finished at: $(date)"
echo ""

###############################################################################
# 実験2: Bank Heist - normal
###############################################################################
TASK="bank_heist"
EXP_NAME="normal"
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
    --agent.dec.simple.img_output normal \
    --jax.platform cuda \
    --logger.outputs jsonl,wandb \
    --logger.videos False \
    2>&1 | tee ${LOG_DIR}/log.log

echo "[Experiment 2] ${TASK} - ${EXP_NAME} Finished at: $(date)"
echo ""

###############################################################################
# 実験3: Frostbite - baseline
###############################################################################
TASK="frostbite"
EXP_NAME="baseline"
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
    --jax.platform cuda \
    --logger.outputs jsonl,wandb \
    --logger.videos False \
    2>&1 | tee ${LOG_DIR}/log.log

echo "[Experiment 3] ${TASK} - ${EXP_NAME} Finished at: $(date)"
echo ""

###############################################################################
# 実験4: Frostbite - normal
###############################################################################
TASK="frostbite"
EXP_NAME="normal"
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
    --agent.dec.simple.img_output normal \
    --jax.platform cuda \
    --logger.outputs jsonl,wandb \
    --logger.videos False \
    2>&1 | tee ${LOG_DIR}/log.log

echo "[Experiment 4] ${TASK} - ${EXP_NAME} Finished at: $(date)"
echo ""

###############################################################################
# 実験5: Hero - baseline
###############################################################################
TASK="hero"
EXP_NAME="baseline"
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
    --jax.platform cuda \
    --logger.outputs jsonl,wandb \
    --logger.videos False \
    2>&1 | tee ${LOG_DIR}/log.log

echo "[Experiment 5] ${TASK} - ${EXP_NAME} Finished at: $(date)"
echo ""

###############################################################################
# 実験6: Hero - normal
###############################################################################
TASK="hero"
EXP_NAME="normal"
TIME_STR=$(date '+%y%m%d%H%M')
LOG_DIR="log/${TIME_STR}_dreamerV3_atari100k_${TASK}_${EXP_NAME}"
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "[Experiment 6] ${TASK} - ${EXP_NAME}"
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
    --agent.dec.simple.img_output normal \
    --jax.platform cuda \
    --logger.outputs jsonl,wandb \
    --logger.videos False \
    2>&1 | tee ${LOG_DIR}/log.log

echo "[Experiment 6] ${TASK} - ${EXP_NAME} Finished at: $(date)"
echo ""

###############################################################################
# 実験7: Kangaroo - baseline
###############################################################################
TASK="kangaroo"
EXP_NAME="baseline"
TIME_STR=$(date '+%y%m%d%H%M')
LOG_DIR="log/${TIME_STR}_dreamerV3_atari100k_${TASK}_${EXP_NAME}"
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "[Experiment 7] ${TASK} - ${EXP_NAME}"
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
    --jax.platform cuda \
    --logger.outputs jsonl,wandb \
    --logger.videos False \
    2>&1 | tee ${LOG_DIR}/log.log

echo "[Experiment 7] ${TASK} - ${EXP_NAME} Finished at: $(date)"
echo ""

###############################################################################
# 実験8: Kangaroo - normal
###############################################################################
TASK="kangaroo"
EXP_NAME="normal"
TIME_STR=$(date '+%y%m%d%H%M')
LOG_DIR="log/${TIME_STR}_dreamerV3_atari100k_${TASK}_${EXP_NAME}"
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "[Experiment 8] ${TASK} - ${EXP_NAME}"
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
    --agent.dec.simple.img_output normal \
    --jax.platform cuda \
    --logger.outputs jsonl,wandb \
    --logger.videos False \
    2>&1 | tee ${LOG_DIR}/log.log

echo "[Experiment 8] ${TASK} - ${EXP_NAME} Finished at: $(date)"
echo ""

###############################################################################
# 実験9: Alien - baseline
###############################################################################
TASK="alien"
EXP_NAME="baseline"
TIME_STR=$(date '+%y%m%d%H%M')
LOG_DIR="log/${TIME_STR}_dreamerV3_atari100k_${TASK}_${EXP_NAME}"
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "[Experiment 9] ${TASK} - ${EXP_NAME}"
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
    --jax.platform cuda \
    --logger.outputs jsonl,wandb \
    --logger.videos False \
    2>&1 | tee ${LOG_DIR}/log.log

echo "[Experiment 9] ${TASK} - ${EXP_NAME} Finished at: $(date)"
echo ""

###############################################################################
# 実験10: Alien - normal
###############################################################################
TASK="alien"
EXP_NAME="normal"
TIME_STR=$(date '+%y%m%d%H%M')
LOG_DIR="log/${TIME_STR}_dreamerV3_atari100k_${TASK}_${EXP_NAME}"
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "[Experiment 10] ${TASK} - ${EXP_NAME}"
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
    --agent.dec.simple.img_output normal \
    --jax.platform cuda \
    --logger.outputs jsonl,wandb \
    --logger.videos False \
    2>&1 | tee ${LOG_DIR}/log.log

echo "[Experiment 10] ${TASK} - ${EXP_NAME} Finished at: $(date)"
echo ""

###############################################################################
# 実験11: Private Eye - baseline
###############################################################################
TASK="private_eye"
EXP_NAME="baseline"
TIME_STR=$(date '+%y%m%d%H%M')
LOG_DIR="log/${TIME_STR}_dreamerV3_atari100k_${TASK}_${EXP_NAME}"
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "[Experiment 11] ${TASK} - ${EXP_NAME}"
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
    --jax.platform cuda \
    --logger.outputs jsonl,wandb \
    --logger.videos False \
    2>&1 | tee ${LOG_DIR}/log.log

echo "[Experiment 11] ${TASK} - ${EXP_NAME} Finished at: $(date)"
echo ""

###############################################################################
# 実験12: Private Eye - normal
###############################################################################
TASK="private_eye"
EXP_NAME="normal"
TIME_STR=$(date '+%y%m%d%H%M')
LOG_DIR="log/${TIME_STR}_dreamerV3_atari100k_${TASK}_${EXP_NAME}"
mkdir -p ${LOG_DIR}

echo "=========================================="
echo "[Experiment 12] ${TASK} - ${EXP_NAME}"
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
    --agent.dec.simple.img_output normal \
    --jax.platform cuda \
    --logger.outputs jsonl,wandb \
    --logger.videos False \
    2>&1 | tee ${LOG_DIR}/log.log

echo "[Experiment 12] ${TASK} - ${EXP_NAME} Finished at: $(date)"
echo ""

###############################################################################
echo "=========================================="
echo "All experiments completed at: $(date)"
echo "=========================================="
