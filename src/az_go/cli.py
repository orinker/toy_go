import torch

from .augment import augment_batch
from .game import load_go_game
from .learner import AZLearner
from .mcts import MCTS
from .net import AZNet
from .selfplay import ReplayBuffer, play_one_selfplay_game
from .utils import a_to_rc, pass_action


def main_train(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    game = load_go_game(board_size=9, komi=7.5)
    C, H, W = game.observation_tensor_shape()
    assert H == 9 and W == 9, "This script is written for 9x9."

    net = AZNet(in_planes=C, board_size=9, channels=args.channels, blocks=args.blocks).to(device)
    learner = AZLearner(net, lr=args.lr, weight_decay=args.wd, device=device)
    rb = ReplayBuffer(capacity=args.buffer)

    print("Starting self-play + training...")
    total_games = 0
    N = 9

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
        for _ in range(steps):
            if len(rb) < args.batch:
                break
            batch = rb.sample(args.batch)
            if args.augment:
                batch = augment_batch(batch, N)
            logs = learner.train_step(batch)
        print(
            f"Epoch {epoch} | games={total_games} | buffer={len(rb)} | loss={logs['loss']:.3f} "
            f"| p={logs['policy_loss']:.3f} v={logs['value_loss']:.3f} H={logs['entropy']:.2f}"
        )

        if args.ckpt:
            torch.save(net.state_dict(), args.ckpt)

    print("Done. Model saved to:", args.ckpt)


def main_play(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    game = load_go_game(board_size=9, komi=7.5)
    C, H, W = game.observation_tensor_shape()
    net = AZNet(in_planes=C, board_size=9, channels=args.channels, blocks=args.blocks).to(device)
    if args.ckpt:
        net.load_state_dict(torch.load(args.ckpt, map_location=device))
    net.eval()

    state = game.new_initial_state()
    mcts = MCTS(game, net, device=device)

    print("Starting a quick self-play (greedy after first moves):")
    ply = 0
    while not state.is_terminal():
        tau = 1.0 if ply < args.temp_moves else 1e-6
        counts, action, _ = mcts.run(state, num_sims=args.mcts_sims, temperature=tau)
        r, c = a_to_rc(action, 9) if action != pass_action(9) else ("pass", "pass")
        visits = sum(counts.values())
        print(
            f"Ply {ply:02d} | player={state.current_player()} "
            f"| action={action} ({r},{c}) | visits={visits}"
        )
        state.apply_action(action)
        ply += 1

    ret = state.returns()
    print("Final returns [black, white]:", ret, "=> winner:", "Black" if ret[0] > 0 else "White")
