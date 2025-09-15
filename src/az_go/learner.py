from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from .net import AZNet
from .selfplay import Sample


class AZLearner:
    def __init__(self, net: AZNet, lr=1e-3, weight_decay=1e-4, device="cpu"):
        self.net = net.to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device

    def train_step(self, batch: List[Sample]) -> Dict[str, float]:
        x = torch.from_numpy(np.stack([s.planes for s in batch], axis=0)).to(self.device)
        target_pi = torch.from_numpy(np.stack([s.pi for s in batch], axis=0)).to(self.device)
        target_v = torch.from_numpy(np.array([s.z for s in batch], dtype=np.float32)).to(self.device)

        logits, v = self.net(x)
        v = v.squeeze(1)

        logp = F.log_softmax(logits, dim=1)
        policy_loss = -(target_pi * logp).sum(dim=1).mean()
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

