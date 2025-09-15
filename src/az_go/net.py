
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.torso(self.stem(x))

        p = self.p_head(h)
        p = p.view(p.size(0), -1)
        logits = self.p_fc(p)

        v = self.v_head(h)
        v = v.view(v.size(0), -1)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v))
        return logits, v

