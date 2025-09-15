import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    children: Dict[int, "Node"] = None
    is_expanded: bool = False
    legal_actions: Optional[List[int]] = None

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

    def run(self, root_state, num_sims: int, temperature: float = 1.0) -> Tuple[Dict[int, int], int, Node]:
        root = Node(prior=1.0, to_play=root_state.current_player())
        self._expand(root, root_state)

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
                black_final = state.returns()[0]
                leaf_value = black_final if node.to_play == 0 else -black_final
            else:
                planes = obs_to_planes(self.game, state)
                x = torch.from_numpy(planes[None, ...]).to(self.device)
                with torch.no_grad():
                    logits, v = self.net(x)
                    v = float(v.item())
                    policy = logits.squeeze(0).cpu().numpy()
                legal = state.legal_actions()
                mask = np.full(self.net.num_actions, -1e9, dtype=np.float32)
                mask[legal] = policy[legal]
                probs = np.exp(mask - mask.max())
                probs = probs / probs.sum()
                self._expand(node, state, prior_over_actions=probs, legal_actions=legal)
                leaf_value = v

            # Backup
            for n in reversed(path):
                n.N += 1
                n.W += leaf_value
                n.Q = n.W / n.N
                leaf_value = -leaf_value

        counts = {a: child.N for a, child in root.children.items()}
        chosen = self._sample_action_from_counts(counts, temperature)
        return counts, chosen, root

    def _add_root_noise(self, root: Node):
        legal = root.legal_actions
        if not legal:
            return
        alpha = self.dirichlet_alpha
        noise = np.random.dirichlet([alpha] * len(legal))
        for a, n in zip(legal, noise):
            root.children[a].prior = (1 - self.dirichlet_eps) * root.children[a].prior + self.dirichlet_eps * float(n)

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
        prior_over_actions: Optional[np.ndarray] = None,
        legal_actions: Optional[List[int]] = None,
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
                node.children[a] = Node(prior=float(prior_over_actions[a]), to_play=1 - node.to_play)
        node.is_expanded = True

    @staticmethod
    def _sample_action_from_counts(counts: Dict[int, int], temperature: float) -> int:
        if temperature <= 1e-6:
            return max(counts.items(), key=lambda kv: kv[1])[0]
        vs = np.array(list(counts.values()), dtype=np.float64)
        ks = np.array(list(counts.keys()), dtype=np.int64)
        exps = np.power(vs, 1.0 / temperature)
        if exps.sum() == 0:
            probs = np.ones_like(exps) / len(exps)
        else:
            probs = exps / exps.sum()
        return int(np.random.choice(ks, p=probs))

