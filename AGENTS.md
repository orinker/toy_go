# Repository Guidelines

This repo contains a minimal AlphaZero-style 9x9 Go implementation using OpenSpiel (`pyspiel`) and PyTorch. The entry point is `main.py`, which exposes `train` and `play` subcommands.

## Project Structure & Module Organization
- `main.py` — model (ResNet), MCTS, self-play, training loop, CLI.
- `checkpoints/` — saved models (create as needed), e.g., `checkpoints/go9_az.pt`.
- `tests/` — Python tests (optional; add as project grows).

## Build, Test, and Development Commands
- Environment: Python 3.10+ with `torch`, `numpy`, and OpenSpiel (`pyspiel`).
- Install (example): `pip install torch numpy` and install OpenSpiel so `import pyspiel` works (follow your platform’s instructions).
- Train: `python3 main.py train --epochs 5 --selfplay_games_per_epoch 10 --ckpt checkpoints/go9_az.pt`
- Quick play: `python3 main.py play --mcts_sims 200 --ckpt checkpoints/go9_az.pt`
- Force CPU: append `--cpu`; CUDA is auto-detected otherwise.

## Coding Style & Naming Conventions
- Python, 4-space indentation, type hints where helpful.
- Naming: functions/variables `snake_case`; classes `CapWords`; constants `UPPER_SNAKE_CASE`.
- Keep modules single-purpose; prefer small, readable functions.
- Formatting/linting (recommended): `black` (line length 100) and `ruff`. Example: `black . && ruff check .`

## Testing Guidelines
- Framework: `pytest`. Place tests in `tests/test_*.py`.
- Run: `pytest -q` (optionally with coverage: `pytest -q --cov=.`).
- Aim for coverage of utilities (e.g., `a_to_rc`, `rc_to_a`, `infer_board_size_from_num_actions`, augmentation). Seed randomness in tests: `np.random.seed(0); torch.manual_seed(0)`.

## Commit & Pull Request Guidelines
- Use Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
- Keep commits atomic with clear scope, e.g., `feat(mcts): tune PUCT constant`.
- PRs should include: purpose, summary of changes, how to run (`train`/`play` commands), sample logs, and any linked issues.

## Security & Configuration Tips
- Do not commit large artifacts; add `checkpoints/*.pt` to `.gitignore`.
- Verify `pyspiel` installation matches your Python/CUDA toolchain; check GPU with `python -c "import torch; print(torch.cuda.is_available())"`.
- Repro notes: consider fixing seeds and recording CLI flags when sharing results.

