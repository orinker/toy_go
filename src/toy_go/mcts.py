import math
from dataclasses import dataclass

import numpy as np
import torch

from .game import obs_to_planes
from .net import AZNet


@dataclass
class Node:
    prior: float
    to_play: int  # 0 = black, 1 = white
    N: int = 0
    W: float = 0.0
    Q: float = 0.0
    children: dict[int, "Node"] = None
    is_expanded: bool = False
    legal_actions: list[int] | None = None
    noise_applied: bool = False

    def __post_init__(self):
        if self.children is None:
            self.children = {}


class MCTS:
    def __init__(
        self,
        game,
        net: AZNet,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        device: str = "cpu",
    ):
        self.game = game
        self.net = net
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.device = device
        self.reset()

    def run(
        self, root_state, num_sims: int, temperature: float = 1.0
    ) -> tuple[dict[int, int], int, Node]:
        root = self._ensure_root(root_state)
        if root_state.is_terminal():
            return {}, -1, root

        was_training = self.net.training
        if was_training:
            self.net.eval()

        try:
            if not root.is_expanded:
                self._evaluate_and_expand(root, root_state)

            self._add_root_noise(root)

            for _ in range(num_sims):
                state = root_state.clone()
                path = [root]
                node = root

                # Selection
                while node.is_expanded and not state.is_terminal():
                    a = self._select_action(node)
                    state.apply_action(a)
                    if a not in node.children:
                        node.children[a] = Node(prior=1e-8, to_play=1 - node.to_play)
                    node = node.children[a]
                    path.append(node)

                # Evaluate/Expand
                if state.is_terminal():
                    result = float(np.sign(state.returns()[0]))
                    leaf_value = result if node.to_play == 0 else -result
                else:
                    leaf_value = self._evaluate_and_expand(node, state)

                # Backup
                for n in reversed(path):
                    n.N += 1
                    n.W += leaf_value
                    n.Q = n.W / n.N
                    leaf_value = -leaf_value

            counts = {a: child.N for a, child in root.children.items()}
            chosen = self._sample_action_from_counts(counts, temperature)
            return counts, chosen, root
        finally:
            if was_training:
                self.net.train()

    def reset(self):
        self.root: Node | None = None
        self.root_history: tuple[int, ...] = ()

    def advance(self, action: int):
        if self.root is None:
            return
        child = self.root.children.get(action)
        if child is None:
            self.reset()
            return
        self.root = child
        self.root_history = self.root_history + (action,)
        self.root.noise_applied = False

    def _ensure_root(self, state) -> Node:
        history = tuple(state.history())
        if self.root is None or self.root_history != history:
            self.root = Node(prior=1.0, to_play=state.current_player())
            self.root_history = history
        return self.root

    def _add_root_noise(self, root: Node):
        if self.dirichlet_eps <= 0 or root.noise_applied:
            return
        legal = root.legal_actions
        if not legal:
            return
        alpha = self.dirichlet_alpha
        noise = np.random.dirichlet([alpha] * len(legal))
        for a, n in zip(legal, noise):
            root.children[a].prior = (
                (1 - self.dirichlet_eps) * root.children[a].prior
                + self.dirichlet_eps * float(n)
            )
        root.noise_applied = True

    def _select_action(self, node: Node) -> int:
        total_N = max(1, sum(child.N for child in node.children.values()))
        best, best_score = None, -1e30
        for a, child in node.children.items():
            U = self.c_puct * child.prior * math.sqrt(total_N) / (1 + child.N)
            score = child.Q + U
            if score > best_score:
                best, best_score = a, score
        return best

    def _expand(
        self,
        node: Node,
        state,
        prior_over_actions: np.ndarray | None = None,
        legal_actions: list[int] | None = None,
    ):
        if node.is_expanded:
            return
        if legal_actions is None:
            legal_actions = state.legal_actions()
        node.legal_actions = legal_actions

        if prior_over_actions is None:
            prior_over_actions = np.zeros(self.net.num_actions, dtype=np.float32)
            prior_over_actions[legal_actions] = 1.0 / max(1, len(legal_actions))

        for a in legal_actions:
            if a not in node.children:
                node.children[a] = Node(
                    prior=float(prior_over_actions[a]),
                    to_play=1 - node.to_play,
                )
        node.is_expanded = True

    def _evaluate_and_expand(self, node: Node, state) -> float:
        value, probs, legal = self._policy_value(state)
        self._expand(node, state, prior_over_actions=probs, legal_actions=legal)
        return value

    def _policy_value(self, state) -> tuple[float, np.ndarray, list[int]]:
        planes = obs_to_planes(self.game, state)
        x = torch.from_numpy(planes[None, ...]).to(self.device)
        with torch.no_grad():
            logits, value = self.net(x)
        policy_logits = logits.squeeze(0).cpu().numpy()
        value = float(value.item())
        legal = state.legal_actions()
        probs = np.zeros(self.net.num_actions, dtype=np.float32)
        if legal:
            legal_logits = policy_logits[legal]
            legal_logits = legal_logits - legal_logits.max()
            exp_logits = np.exp(legal_logits.astype(np.float64))
            total = exp_logits.sum()
            if total <= 0:
                probs[legal] = 1.0 / len(legal)
            else:
                probs[legal] = (exp_logits / total).astype(np.float32)
        return value, probs, legal

    @staticmethod
    def _sample_action_from_counts(counts: dict[int, int], temperature: float) -> int:
        if temperature <= 1e-6:
            return max(counts.items(), key=lambda kv: kv[1])[0]
        vs = np.array(list(counts.values()), dtype=np.float64)
        ks = np.array(list(counts.keys()), dtype=np.int64)
        if np.all(vs == 0):
            probs = np.ones_like(vs) / len(vs)
        else:
            positive = vs > 0
            log_vs = np.full_like(vs, -np.inf)
            log_vs[positive] = np.log(vs[positive]) / temperature
            max_log = np.max(log_vs[positive])
            stabilized = np.exp(log_vs - max_log)
            total = stabilized.sum()
            if not np.isfinite(total) or total <= 0:
                probs = np.ones_like(stabilized) / len(stabilized)
            else:
                probs = stabilized / total
        return int(np.random.choice(ks, p=probs))
