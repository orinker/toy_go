from pathlib import Path

import torch

from .augment import augment_batch
from .game import load_go_game
from .learner import AZLearner
from .mcts import MCTS
from .net import AZNet
from .selfplay import ReplayBuffer, play_one_selfplay_game
from .utils import a_to_rc, pass_action


def _load_model_state(net: AZNet, ckpt_path: Path, device: str):
    payload = torch.load(ckpt_path, map_location=device)
    if isinstance(payload, dict) and "model" in payload:
        net_state = payload["model"]
        opt_state = payload.get("optimizer")
    else:
        net_state = payload
        opt_state = None
    net.load_state_dict(net_state)
    return opt_state


def _load_training_checkpoint(net: AZNet, learner: AZLearner, ckpt_path: Path, device: str) -> None:
    opt_state = _load_model_state(net, ckpt_path, device)
    if opt_state is not None:
        learner.opt.load_state_dict(opt_state)


def main_train(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    board_size = args.board_size
    komi = args.komi
    game = load_go_game(board_size=board_size, komi=komi)
    C, H, W = game.observation_tensor_shape()
    if H != board_size or W != board_size:
        raise ValueError(
            f"Observation shape mismatch: expected {board_size}x{board_size}, got {H}x{W}"
        )

    net = AZNet(
        in_planes=C, board_size=board_size, channels=args.channels, blocks=args.blocks
    ).to(device)
    learner = AZLearner(net, lr=args.lr, weight_decay=args.wd, device=device)
    rb = ReplayBuffer(capacity=args.buffer)

    ckpt_path = Path(args.ckpt).expanduser() if args.ckpt else None
    if args.resume:
        if ckpt_path is None:
            raise ValueError("--continue requires --ckpt to point to an existing checkpoint")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        _load_training_checkpoint(net, learner, ckpt_path, device)
        print(f"Loaded checkpoint from {ckpt_path}")

    print("Starting self-play + training...")
    total_games = 0
    N = board_size

    for epoch in range(1, args.epochs + 1):
        # Self-play
        for _ in range(args.selfplay_games_per_epoch):
            samples = play_one_selfplay_game(
                game=game,
                net=net,
                mcts_sims=args.mcts_sims,
                temp_moves=args.temp_moves,
                device=device,
            )
            for s in samples:
                rb.push(s)
            total_games += 1

        # Training
        steps = args.updates_per_epoch
        logs: dict[str, float] | None = None
        for _ in range(steps):
            if len(rb) < args.batch:
                break
            batch = rb.sample(args.batch)
            if args.augment:
                batch = augment_batch(batch, N)
            logs = learner.train_step(batch)
        if logs is None:
            print(
                f"Epoch {epoch} | games={total_games} | buffer={len(rb)} | loss=-- | "
                f"p=-- v=-- H=--"
            )
        else:
            print(
                f"Epoch {epoch} | games={total_games} | buffer={len(rb)} | loss={logs['loss']:.3f} "
                f"| p={logs['policy_loss']:.3f} v={logs['value_loss']:.3f} H={logs['entropy']:.2f}"
            )

        if ckpt_path is not None:
            if not ckpt_path.parent.exists():
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": net.state_dict(),
                    "optimizer": learner.opt.state_dict(),
                },
                str(ckpt_path),
            )

    if ckpt_path is not None:
        print("Done. Model saved to:", ckpt_path)
    else:
        print("Done.")


def main_play(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    board_size = args.board_size
    komi = args.komi
    game = load_go_game(board_size=board_size, komi=komi)
    C, H, W = game.observation_tensor_shape()
    if H != board_size or W != board_size:
        raise ValueError(
            f"Observation shape mismatch: expected {board_size}x{board_size}, got {H}x{W}"
        )
    net = AZNet(
        in_planes=C, board_size=board_size, channels=args.channels, blocks=args.blocks
    ).to(device)
    if args.ckpt:
        ckpt_path = Path(args.ckpt).expanduser()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        _load_model_state(net, ckpt_path, device)
    net.eval()

    state = game.new_initial_state()
    mcts = MCTS(game, net, device=device)

    print("Starting a quick self-play (greedy after first moves):")
    ply = 0
    while not state.is_terminal():
        tau = 1.0 if ply < args.temp_moves else 1e-6
        counts, action, _ = mcts.run(state, num_sims=args.mcts_sims, temperature=tau)
        r, c = (
            a_to_rc(action, board_size)
            if action != pass_action(board_size)
            else ("pass", "pass")
        )
        visits = sum(counts.values())
        print(
            f"Ply {ply:02d} | player={state.current_player()} "
            f"| action={action} ({r},{c}) | visits={visits}"
        )
        state.apply_action(action)
        ply += 1

    ret = state.returns()
    print("Final returns [black, white]:", ret, "=> winner:", "Black" if ret[0] > 0 else "White")
