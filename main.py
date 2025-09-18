"""
Thin CLI wrapper for the AlphaZero 9x9 Go prototype.
Actual implementation lives in src/toy_go/* modules.
"""

import argparse
import os
import sys

# Ensure src/ is importable without installing the package
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)



def _run_train(args):
    # Lazy import to avoid heavy deps at import time (e.g., torch, pyspiel)
    from toy_go.cli import main_train as _main_train

    _main_train(args)


def _run_play(args):
    from toy_go.cli import main_play as _main_play

    _main_play(args)


def _run_visualize(args):
    from toy_go.visualize import main_visualize as _main_visualize

    _main_visualize(args)


def _run_pvc(args):
    from toy_go.pvc import main_pvc as _main_pvc

    _main_pvc(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--selfplay_games_per_epoch", type=int, default=10)
    p_train.add_argument("--mcts_sims", type=int, default=160)
    p_train.add_argument("--temp_moves", type=int, default=10)
    p_train.add_argument("--updates_per_epoch", type=int, default=200)
    p_train.add_argument("--batch", type=int, default=128)
    p_train.add_argument("--buffer", type=int, default=50000)
    p_train.add_argument("--channels", type=int, default=64)
    p_train.add_argument("--blocks", type=int, default=6)
    p_train.add_argument("--board_size", type=int, default=9)
    p_train.add_argument("--komi", type=float, default=2.5)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--wd", type=float, default=1e-4)
    p_train.add_argument("--augment", action="store_true")
    p_train.add_argument("--ckpt", type=str, default="go9_az.pt")
    p_train.add_argument("--cpu", action="store_true")
    p_train.add_argument(
        "--continue",
        dest="resume",
        action="store_true",
        help="Continue training from an existing checkpoint",
    )
    p_train.set_defaults(func=_run_train)

    p_play = sub.add_parser("play")
    p_play.add_argument("--mcts_sims", type=int, default=200)
    p_play.add_argument("--temp_moves", type=int, default=10)
    p_play.add_argument("--channels", type=int, default=64)
    p_play.add_argument("--blocks", type=int, default=6)
    p_play.add_argument("--board_size", type=int, default=9)
    p_play.add_argument("--komi", type=float, default=2.5)
    p_play.add_argument("--ckpt", type=str, default="go9_az.pt")
    p_play.add_argument("--cpu", action="store_true")
    p_play.set_defaults(func=_run_play)

    p_visualize = sub.add_parser("visualize")
    p_visualize.add_argument("--mcts_sims", type=int, default=100)
    p_visualize.add_argument("--channels", type=int, default=32)
    p_visualize.add_argument("--blocks", type=int, default=3)
    p_visualize.add_argument("--board_size", type=int, default=9)
    p_visualize.add_argument("--komi", type=float, default=2.5)
    p_visualize.add_argument("--ckpt", type=str, default="checkpoints/1k_go9_az.pt")
    p_visualize.add_argument("--cpu", action="store_true")
    p_visualize.set_defaults(func=_run_visualize)

    p_pvc = sub.add_parser("pvc")
    p_pvc.add_argument("--mcts_sims", type=int, default=200)
    p_pvc.add_argument("--channels", type=int, default=64)
    p_pvc.add_argument("--blocks", type=int, default=6)
    p_pvc.add_argument("--board_size", type=int, default=9)
    p_pvc.add_argument("--komi", type=float, default=2.5)
    p_pvc.add_argument("--ckpt", type=str, default="go9_az.pt")
    p_pvc.add_argument("--cpu", action="store_true")
    p_pvc.set_defaults(func=_run_pvc)

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
