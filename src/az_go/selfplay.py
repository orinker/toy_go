from collections import deque, namedtuple
from typing import Dict, List, Tuple

import numpy as np

from .game import obs_to_planes
from .mcts import MCTS
from .net import AZNet
from .utils import infer_board_size_from_num_actions


Sample = namedtuple("Sample", ["planes", "pi", "z"])


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buf = deque(maxlen=capacity)

    def push(self, s: Sample):
        self.buf.append(s)

    def sample(self, batch_size: int) -> List[Sample]:
        idx = np.random.choice(len(self.buf), size=batch_size, replace=False)
        return [self.buf[i] for i in idx]

    def __len__(self):
        return len(self.buf)


def mcts_policy_from_counts(counts: Dict[int, int], num_actions: int, temperature: float) -> np.ndarray:
    pi = np.zeros(num_actions, dtype=np.float32)
    for a, n in counts.items():
        pi[a] = n
    if temperature <= 1e-6:
        a_star = int(np.argmax(pi))
        out = np.zeros_like(pi)
        out[a_star] = 1.0
        return out
    pi = np.power(pi, 1.0 / temperature)
    if pi.sum() == 0:
        pi[:] = 1.0 / len(pi)
    else:
        pi /= pi.sum()
    return pi


def play_one_selfplay_game(
    game, net: AZNet, mcts_sims: int, temp_moves: int, device: str = "cpu"
) -> List[Sample]:
    """Plays one self-play game using MCTS-guided moves."""
    state = game.new_initial_state()
    N = infer_board_size_from_num_actions(game.num_distinct_actions())
    mcts = MCTS(game, net, device=device)

    traj: List[Tuple[np.ndarray, np.ndarray, int]] = []  # (planes, pi, current_player)
    ply = 0
    while not state.is_terminal():
        tau = 1.0 if ply < temp_moves else 1e-6
        counts, action, _ = mcts.run(state, num_sims=mcts_sims, temperature=tau)
        pi = mcts_policy_from_counts(counts, net.num_actions, tau)

        planes = obs_to_planes(game, state)  # (C,N,N)
        traj.append((planes.copy(), pi.copy(), state.current_player()))

        state.apply_action(action)
        ply += 1

    final_black = state.returns()[0]  # +1 black win, -1 black loss (zero-sum)
    samples: List[Sample] = []
    for planes, pi, cur in traj:
        z = final_black if cur == 0 else -final_black
        samples.append(Sample(planes=planes, pi=pi, z=float(z)))
    return samples

