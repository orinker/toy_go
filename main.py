# az_go9.py
# Reference AlphaZero for 9x9 Go using OpenSpiel + PyTorch
# --------------------------------------------------------
# MIT-style educational sample. Clarity first; small and readable.

import math, random, time, argparse
from collections import deque, defaultdict, namedtuple
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pyspiel  # OpenSpiel Python API

# ---------------------------
# 1) Game / observation utils
# ---------------------------

def load_go_game(board_size=9, komi=7.5):
    # You can also call: pyspiel.load_game("go", {"board_size": board_size, "komi": komi})
    return pyspiel.load_game(f"go(board_size={board_size},komi={komi})")

def infer_board_size_from_num_actions(num_actions: int) -> int:
    # For Go in OpenSpiel the action space is N*N + 1 (the +1 is pass).
    # We assert that here; if it fails, adjust mapping below.
    # (Go appears in the official game list; OpenSpiel uses integer action ids.)  # docs
    # https://openspiel.readthedocs.io/
    n2_plus_pass = num_actions
    n2 = n2_plus_pass - 1
    n = int(round(math.sqrt(n2)))
    if n * n + 1 != num_actions:
        raise ValueError(f"Unexpected action count {num_actions}; expected N*N+1.")
    return n

def obs_to_planes(game, state) -> np.ndarray:
    """
    Convert OpenSpiel observation tensor to (C,H,W) float32 for CNN input.

    OpenSpiel provides game-specific 2D observation tensors for board games that
    are ‘friendly to conv nets’. We query the shape and reshape accordingly. 
    (If the tensor is already flat, we reshape using the provided shape.)
    """
    obs = np.array(state.observation_tensor(), dtype=np.float32)
    shape = game.observation_tensor_shape()
    if len(shape) == 3:
        c, h, w = shape  # OpenSpiel conv models assume a 3D tensor.  (docs: AlphaZero page)
        planes = obs.reshape(c, h, w)
        return planes
    elif len(shape) == 1:
        # Fallback: try to guess 3D from board size and expected planes.
        # For Go, OpenSpiel typically supplies planar inputs (e.g., history planes + to-play).
        # If your build differs, print(shape) and reshape appropriately.
        raise RuntimeError(f"Got flat obs of length {shape[0]}; please reshape per your build.")
    else:
        raise RuntimeError(f"Unexpected observation shape: {shape}")

# Action <-> (row, col) helpers (pass is index N*N).
def a_to_rc(a: int, N: int) -> Tuple[int,int]:
    return (a // N, a % N)

def rc_to_a(r: int, c: int, N: int) -> int:
    return r * N + c

def pass_action(N: int) -> int:
    return N * N

# ------------------------------------
# 2) AlphaZero network (small ResNet)
# ------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(ch)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(x + out)

class AZNet(nn.Module):
    def __init__(self, in_planes: int, board_size: int, channels: int = 64, blocks: int = 6):
        super().__init__()
        self.N = board_size
        self.num_actions = self.N * self.N + 1

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        # Torso
        self.torso = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

        # Policy head
        self.p_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )
        self.p_fc = nn.Linear(2 * self.N * self.N, self.num_actions)

        # Value head
        self.v_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.v_fc1 = nn.Linear(1 * self.N * self.N, 128)
        self.v_fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, C, N, N]
        returns:
          logits: [B, N*N+1] (mask/softmax happens outside)
          value:  [B, 1]     (tanh)
        """
        h = self.torso(self.stem(x))

        p = self.p_head(h)
        p = p.view(p.size(0), -1)
        logits = self.p_fc(p)

        v = self.v_head(h)
        v = v.view(v.size(0), -1)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v))
        return logits, v

# -------------------------------
# 3) MCTS with PUCT + root noise
# -------------------------------

@dataclass
class Node:
    prior: float
    to_play: int                 # 0 = black, 1 = white
    N: int = 0                   # visit count
    W: float = 0.0               # total value from this player's perspective
    Q: float = 0.0               # mean value
    children: Dict[int, 'Node'] = None  # action -> Node
    is_expanded: bool = False
    legal_actions: Optional[List[int]] = None

    def __post_init__(self):
        if self.children is None:
            self.children = {}

class MCTS:
    def __init__(self, game, net: AZNet, c_puct: float = 1.5, dirichlet_alpha: float = 0.3, dirichlet_eps: float = 0.25, device="cpu"):
        self.game = game
        self.net = net
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.device = device

    def run(self, root_state, num_sims: int, temperature: float = 1.0) -> Tuple[Dict[int,int], int, Node]:
        """
        Returns:
          visit_counts: dict action->N
          chosen_action: sampled according to temperature-adjusted counts
          root: root node (for debugging/analysis)
        """
        root = Node(prior=1.0, to_play=root_state.current_player())
        self._expand(root, root_state)

        # Dirichlet noise at root to encourage exploration
        self._add_root_noise(root)

        for _ in range(num_sims):
            state = root_state.clone()
            path = [root]
            node = root

            # SELECTION
            while node.is_expanded and not state.is_terminal():
                a = self._select_action(node)
                state.apply_action(a)
                if a not in node.children:
                    # Should not happen if we always expand on first visit
                    node.children[a] = Node(prior=1e-8, to_play=1 - node.to_play)
                node = node.children[a]
                path.append(node)

            # EVALUATE / EXPAND
            if state.is_terminal():
                # Terminal value from the perspective of 'node.to_play' at this node:
                # returns()[0] is Black's final return (zero-sum).
                black_final = state.returns()[0]
                leaf_value = black_final if node.to_play == 0 else -black_final
            else:
                # Evaluate network at leaf for current player to move
                planes = obs_to_planes(self.game, state)
                x = torch.from_numpy(planes[None, ...]).to(self.device)  # [1, C, N, N]
                with torch.no_grad():
                    logits, v = self.net(x)
                    v = float(v.item())
                    policy = logits.squeeze(0).cpu().numpy()
                # Mask illegal moves and renormalize
                legal = state.legal_actions()
                mask = np.full(self.net.num_actions, -1e9, dtype=np.float32)
                mask[legal] = policy[legal]
                probs = np.exp(mask - mask.max())
                probs = probs / probs.sum()
                self._expand(node, state, prior_over_actions=probs, legal_actions=legal)
                leaf_value = v  # v is already from the current player's perspective

            # BACKUP
            # Alternate signs as we go up the tree: parent stores value for its 'to_play'
            for n in reversed(path):
                n.N += 1
                n.W += leaf_value
                n.Q = n.W / n.N
                leaf_value = -leaf_value  # flip perspective

        # ACTION FROM VISIT COUNTS
        counts = {a: child.N for a, child in root.children.items()}
        chosen = self._sample_action_from_counts(counts, temperature)
        return counts, chosen, root

    def _add_root_noise(self, root: Node):
        legal = root.legal_actions
        if not legal: return
        alpha = self.dirichlet_alpha
        noise = np.random.dirichlet([alpha] * len(legal))
        for a, n in zip(legal, noise):
            root.children[a].prior = (1 - self.dirichlet_eps) * root.children[a].prior + self.dirichlet_eps * float(n)

    def _select_action(self, node: Node) -> int:
        # PUCT: select argmax_a Q + U, with U ∝ prior * sqrt(sumN) / (1+N_a)
        total_N = max(1, sum(child.N for child in node.children.values()))
        best, best_score = None, -1e30
        for a, child in node.children.items():
            U = self.c_puct * child.prior * math.sqrt(total_N) / (1 + child.N)
            score = child.Q + U
            if score > best_score:
                best, best_score = a, score
        return best

    def _expand(self, node: Node, state, prior_over_actions: Optional[np.ndarray] = None, legal_actions: Optional[List[int]] = None):
        if node.is_expanded: return
        if legal_actions is None:
            legal_actions = state.legal_actions()
        node.legal_actions = legal_actions

        if prior_over_actions is None:
            # Uniform priors if net not evaluated yet (only at root on first call)
            prior_over_actions = np.zeros(self.net.num_actions, dtype=np.float32)
            prior_over_actions[legal_actions] = 1.0 / max(1, len(legal_actions))

        for a in legal_actions:
            if a not in node.children:
                node.children[a] = Node(prior=float(prior_over_actions[a]), to_play=1 - node.to_play)
        node.is_expanded = True

    @staticmethod
    def _sample_action_from_counts(counts: Dict[int,int], temperature: float) -> int:
        if temperature <= 1e-6:
            # Greedy
            return max(counts.items(), key=lambda kv: kv[1])[0]
        vs = np.array(list(counts.values()), dtype=np.float64)
        ks = np.array(list(counts.keys()), dtype=np.int64)
        # Soft sampling over visit counts^{1/tau}
        exps = np.power(vs, 1.0 / temperature)
        if exps.sum() == 0:
            # All zero (rare early), fallback to uniform
            probs = np.ones_like(exps) / len(exps)
        else:
            probs = exps / exps.sum()
        return int(np.random.choice(ks, p=probs))

# ---------------------------------
# 4) Self-play + replay buffer
# ---------------------------------

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

def mcts_policy_from_counts(counts: Dict[int,int], num_actions: int, temperature: float) -> np.ndarray:
    # Build a probability vector aligned to action ids
    pi = np.zeros(num_actions, dtype=np.float32)
    for a, n in counts.items():
        pi[a] = n
    if temperature <= 1e-6:
        # One-hot on argmax visit
        a_star = int(np.argmax(pi))
        out = np.zeros_like(pi)
        out[a_star] = 1.0
        return out
    # Soft with temperature
    pi = np.power(pi, 1.0 / temperature)
    if pi.sum() == 0:
        pi[:] = 1.0 / len(pi)
    else:
        pi /= pi.sum()
    return pi

def play_one_selfplay_game(game, net: AZNet, mcts_sims: int, temp_moves: int, device="cpu") -> List[Sample]:
    """
    Plays one self-play game using MCTS-guided moves.
    temp_moves: for the first k plies, sample with temperature=1.0; afterwards set tau≈0.
    """
    state = game.new_initial_state()
    N = infer_board_size_from_num_actions(game.num_distinct_actions())
    mcts = MCTS(game, net, device=device)

    traj: List[Tuple[np.ndarray, np.ndarray, int]] = []  # (planes, pi, current_player)
    ply = 0
    while not state.is_terminal():
        tau = 1.0 if ply < temp_moves else 1e-6
        counts, action, root = mcts.run(state, num_sims=mcts_sims, temperature=tau)
        pi = mcts_policy_from_counts(counts, net.num_actions, tau)

        planes = obs_to_planes(game, state)  # (C,N,N)
        traj.append((planes.copy(), pi.copy(), state.current_player()))

        state.apply_action(action)
        ply += 1

    # Terminal outcome: OpenSpiel returns per-player returns at terminal.
    final_black = state.returns()[0]  # +1 black win, -1 black loss (zero-sum)
    samples: List[Sample] = []
    for planes, pi, cur in traj:
        z = final_black if cur == 0 else -final_black
        samples.append(Sample(planes=planes, pi=pi, z=float(z)))
    return samples

# ------------------------
# 5) Training / evaluation
# ------------------------

class AZLearner:
    def __init__(self, net: AZNet, lr=1e-3, weight_decay=1e-4, device="cpu"):
        self.net = net.to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device

    def train_step(self, batch: List[Sample]) -> Dict[str, float]:
        x = torch.from_numpy(np.stack([s.planes for s in batch], axis=0)).to(self.device)  # [B,C,N,N]
        target_pi = torch.from_numpy(np.stack([s.pi for s in batch], axis=0)).to(self.device)  # [B,A]
        target_v  = torch.from_numpy(np.array([s.z for s in batch], dtype=np.float32)).to(self.device)  # [B]

        logits, v = self.net(x)                # logits: [B,A], v: [B,1]
        v = v.squeeze(1)                       # [B]

        # Policy loss: CE(target_pi, log_softmax(logits))
        logp = F.log_softmax(logits, dim=1)
        policy_loss = -(target_pi * logp).sum(dim=1).mean()

        # Value loss: MSE
        value_loss = F.mse_loss(v, target_v)

        loss = policy_loss + value_loss

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        with torch.no_grad():
            entropy = -(F.softmax(logits, dim=1) * logp).sum(dim=1).mean().item()
        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy),
        }

# -----------------------------
# 6) Symmetry augmentation (opt)
# -----------------------------

def rotate_flip_planes(planes: np.ndarray, k_rot: int, flip: bool) -> np.ndarray:
    # planes: (C,N,N)
    out = np.rot90(planes, k=k_rot, axes=(1,2)).copy()
    if flip:
        out = np.flip(out, axis=2).copy()
    return out

def rotate_flip_policy(pi: np.ndarray, N: int, k_rot: int, flip: bool) -> np.ndarray:
    A = N*N + 1
    pi_board = pi[:N*N].reshape(N, N)
    pi_pass  = pi[N*N]
    out = np.rot90(pi_board, k=k_rot, axes=(0,1))
    if flip:
        out = np.flip(out, axis=1)
    out = out.reshape(N*N)
    out_pi = np.zeros_like(pi)
    out_pi[:N*N] = out
    out_pi[N*N] = pi_pass
    return out_pi

def augment_batch(batch: List[Sample], N: int) -> List[Sample]:
    aug = []
    for s in batch:
        k = np.random.randint(0, 4)
        f = bool(np.random.randint(0, 2))
        P = rotate_flip_planes(s.planes, k, f)
        pi = rotate_flip_policy(s.pi, N, k, f)
        aug.append(Sample(P, pi, s.z))
    return aug

# -----------------------------
# 7) Training loop entry points
# -----------------------------

def main_train(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    game = load_go_game(board_size=9, komi=7.5)
    # Observe shape for channel count
    tmp_state = game.new_initial_state()
    C, H, W = game.observation_tensor_shape()
    assert H == 9 and W == 9, "This script is written for 9x9."

    net = AZNet(in_planes=C, board_size=9, channels=args.channels, blocks=args.blocks).to(device)
    learner = AZLearner(net, lr=args.lr, weight_decay=args.wd, device=device)
    rb = ReplayBuffer(capacity=args.buffer)

    print("Starting self-play + training...")
    total_games = 0
    N = 9

    for epoch in range(1, args.epochs + 1):
        # ----- Self-play -----
        for _ in range(args.selfplay_games_per_epoch):
            samples = play_one_selfplay_game(
                game=game, net=net,
                mcts_sims=args.mcts_sims,
                temp_moves=args.temp_moves,
                device=device
            )
            for s in samples:
                rb.push(s)
            total_games += 1

        # ----- Training -----
        steps = args.updates_per_epoch
        for step in range(steps):
            if len(rb) < args.batch:
                break
            batch = rb.sample(args.batch)
            if args.augment:
                batch = augment_batch(batch, N)
            logs = learner.train_step(batch)
        print(f"Epoch {epoch} | games={total_games} | buffer={len(rb)} | loss={logs['loss']:.3f} "
              f"| p={logs['policy_loss']:.3f} v={logs['value_loss']:.3f} H={logs['entropy']:.2f}")

        # Save checkpoint
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
        print(f"Ply {ply:02d} | player={state.current_player()} | action={action} ({r},{c}) | visits={sum(counts.values())}")
        state.apply_action(action)
        ply += 1

    ret = state.returns()
    print("Final returns [black, white]:", ret, "=> winner:", "Black" if ret[0] > 0 else "White")

if __name__ == "__main__":
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
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--wd", type=float, default=1e-4)
    p_train.add_argument("--augment", action="store_true")
    p_train.add_argument("--ckpt", type=str, default="go9_az.pt")
    p_train.add_argument("--cpu", action="store_true")
    p_train.set_defaults(func=main_train)

    p_play = sub.add_parser("play")
    p_play.add_argument("--mcts_sims", type=int, default=200)
    p_play.add_argument("--temp_moves", type=int, default=10)
    p_play.add_argument("--channels", type=int, default=64)
    p_play.add_argument("--blocks", type=int, default=6)
    p_play.add_argument("--ckpt", type=str, default="go9_az.pt")
    p_play.add_argument("--cpu", action="store_true")
    p_play.set_defaults(func=main_play)

    args = parser.parse_args()
    args.func(args)
