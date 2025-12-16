# Repository Guidelines

Concise guide for contributors working on this DreamerV3 JAX implementation. Keep changes small, follow existing patterns, and prioritize reproducibility.

## Project Structure & Modules
- `dreamerv3/`: Agent logic (`agent.py`), RSSM (`rssm.py`), configs (`configs.yaml`), entrypoint (`main.py`).
- `embodied/`: Generic RL infra (driver, replay, logger), environment adapters under `embodied/envs/`, utilities in `embodied/core/` and `embodied/jax/`.
- `embodied/tests/`: Pytest suite. Logs and checkpoints are expected under `log/` or `~/logdir`.
- Scripts: `run_experiment.sh` (example training wrapper), `run_nohup.sh`, `Dockerfile` for containerized runs.

## Setup, Build, and Run
- Python 3.11+. Prefer UV for speed:
  ```sh
  uv venv && source .venv/bin/activate
  uv pip install -e ".[all,dev]"
  ```
- Minimal install (no extra envs): `uv pip install -e .` or `pip install -U -r requirements.txt`.
- Train example (GPU default):  
  `python dreamerv3/main.py --configs crafter --logdir ~/logdir/dreamer/{timestamp}`
- Debug small model: add `--configs debug`. CPU run: `--jax.platform cpu`.
- Scope viewer for metrics: `python -m scope.viewer --basedir ~/logdir --port 8000`.

## Coding Style & Conventions
- Python, 2-space indents as in existing files; keep imports standard/library/third-party/local.
- Favor pure functions and small helpers; reuse config utilities in `elements` instead of ad-hoc parsing.
- Docstrings when adding public helpers; Google-style preferred. Keep logging minimal and actionable.
- Configs live in `dreamerv3/configs.yaml`; prefer overriding via flags rather than hardcoding.

## Testing Guidelines
- Run full suite: `pytest embodied/tests`. Target CPU where possible; GPU-specific issues should be gated behind flags.
- Name tests `test_*` and place under `embodied/tests/`; mirror module names when adding new coverage.
- For long runs, scope with `-k pattern` or run single test, e.g., `pytest embodied/tests/test_train.py::TestTrain::test_run_loop`.

## Commit & Pull Request Guidelines
- Commit messages: short, imperative, <72 chars (e.g., “Fix parallel driver argument”).
- PRs should include: summary, key command used, config/seed changes, and before/after metrics or logdir links; add screenshots only when UI changes occur.
- Keep diffs small; note any user-facing behavior changes and new dependencies.

## Security & Configuration Tips
- Store secrets (e.g., `WANDB_API_KEY`) in `.env` or environment variables; do not commit them.
- Verify CUDA/JAX compatibility; for CI or CPU-only runs, force `--jax.platform cpu`.
- Large logdirs/checkpoints belong outside the repo (`log/` is fine but keep it git-ignored).
