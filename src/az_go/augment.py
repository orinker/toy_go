from typing import List

import numpy as np

from .selfplay import Sample


def rotate_flip_planes(planes: np.ndarray, k_rot: int, flip: bool) -> np.ndarray:
    out = np.rot90(planes, k=k_rot, axes=(1, 2)).copy()
    if flip:
        out = np.flip(out, axis=2).copy()
    return out


def rotate_flip_policy(pi: np.ndarray, N: int, k_rot: int, flip: bool) -> np.ndarray:
    pi_board = pi[: N * N].reshape(N, N)
    pi_pass = pi[N * N]
    out = np.rot90(pi_board, k=k_rot, axes=(0, 1))
    if flip:
        out = np.flip(out, axis=1)
    out = out.reshape(N * N)
    out_pi = np.zeros_like(pi)
    out_pi[: N * N] = out
    out_pi[N * N] = pi_pass
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

