# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

DreamerV3 is a scalable reinforcement learning algorithm that learns a world model from experiences and trains an actor-critic policy from imagined trajectories. This implementation is based on JAX and uses the Ninjax library for neural network modules.

## Core Architecture

### Three-Layer Structure

1. **DreamerV3 Agent** (`dreamerv3/agent.py`)
   - World model components: encoder, dynamics (RSSM), decoder
   - Prediction heads: reward, continuation, policy, value
   - Handles training and policy execution

2. **Embodied Library** (`embodied/`)
   - Generic RL infrastructure independent of DreamerV3
   - Core components: Driver, Replay, Logger, Environment wrappers
   - JAX utilities: neural network layers, optimizers, transformations

3. **Environment Adapters** (`embodied/envs/`)
   - Adapters for multiple RL environments (Atari, DMC, Crafter, Minecraft, etc.)
   - Each adapter normalizes the environment to a common interface

### Key Components

**RSSM (Recurrent State-Space Model)** (`dreamerv3/rssm.py`)
- Core world model implementing the dynamics
- Maintains deterministic and stochastic state representations
- Uses categorical distributions for stochastic states

**Configuration System** (`dreamerv3/configs.yaml`)
- All hyperparameters and environment settings
- Config blocks can be combined (e.g., `--configs atari size50m`)
- Includes model size presets (1M to 400M parameters)

**Training Scripts** (`embodied/run/`)
- `train.py`: Single-process training
- `train_eval.py`: Training with separate evaluation
- `parallel.py`: Distributed training across multiple processes

## Common Commands

### Training

Basic training:
```sh
python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/{timestamp} \
  --configs crafter \
  --run.train_ratio 32
```

Train on specific task:
```sh
python dreamerv3/main.py \
  --logdir ~/logdir/atari_pong \
  --configs atari \
  --task atari_pong
```

Debug mode (smaller networks, faster logging):
```sh
python dreamerv3/main.py \
  --logdir ~/logdir/debug \
  --configs crafter debug
```

Resume training (point to same logdir):
```sh
python dreamerv3/main.py \
  --logdir ~/logdir/existing_run \
  --configs crafter
```

### Platform Selection

Switch between GPU/CPU/TPU:
```sh
# CPU
--jax.platform cpu

# GPU (default)
--jax.platform cuda

# TPU
--jax.platform tpu
```

### Viewing Results

```sh
pip install -U scope
python -m scope.viewer --basedir ~/logdir --port 8000
```

Metrics are also saved as JSONL in `{logdir}/metrics.jsonl`.

### Testing

Run tests:
```sh
pytest embodied/tests/
```

Run specific test:
```sh
pytest embodied/tests/test_train.py::TestTrain::test_run_loop
```

## Configuration System

All config options are in `dreamerv3/configs.yaml`. Override from command line:

```sh
# Single option
--batch_size 32

# Nested option
--env.atari.size [128,128]

# Multiple config blocks (later ones override earlier)
--configs atari size50m debug
```

Important config blocks:
- `debug`: Fast debugging (small networks, frequent logs)
- `size{1m,12m,25m,50m,100m,200m,400m}`: Model size presets
- Task-specific: `atari`, `dmc_vision`, `dmc_proprio`, `crafter`, `minecraft`

## Code Organization

### Agent Training Flow

1. `main.py` → Parses config, creates agent/replay/env/logger
2. `embodied.run.train()` → Main training loop
3. `Driver` → Manages parallel environments, collects experiences
4. `Replay` → Stores and samples batches
5. `Agent.train()` → Updates world model and policy
6. `Logger` → Records metrics

### World Model Training

The agent trains a world model with these components:
- **Encoder**: Converts observations to tokens
- **RSSM Dynamics**: Predicts future latent states
- **Decoder**: Reconstructs observations from latent states
- **Reward Head**: Predicts rewards
- **Continuation Head**: Predicts episode termination

### Policy Training

Uses imagined trajectories from the world model:
- Roll out sequences in imagination
- Compute value targets using λ-returns
- Optimize policy with advantage-weighted regression
- Update value function to match returns

## JAX and Ninjax

This codebase uses Ninjax, a JAX neural network library:
- Modules inherit from `nj.Module`
- Parameters are managed automatically via context
- Use `nj.scan` for efficient RNN unrolling
- Training uses `jax.grad` and `jax.jit` for performance

## Common Pitfalls

**"Too many leaves for PyTreeDef" error**: Checkpoint incompatible with current config. Usually happens when reusing an old logdir with different hyperparameters. Delete checkpoint or use a new logdir.

**CUDA errors**: Often preceded by the real error (OOM, version mismatch). Try `--batch_size 1` to test for OOM issues.

**Platform issues**: GPU unavailable → Use `--jax.platform cpu` to run on CPU

**Config precedence**: Config blocks are applied left to right. `--configs defaults atari debug` means debug settings override atari settings.

## Development Workflow

1. For quick iteration, use `debug` config block
2. Test changes with simple environments first (dummy, crafter)
3. Use `--batch_size 1` to isolate shape/NaN issues from OOM
4. Check `{logdir}/config.yaml` to verify applied configuration
5. Monitor `metrics.jsonl` or use Scope viewer for live metrics
