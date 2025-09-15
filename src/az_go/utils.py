import math
from typing import Tuple


def infer_board_size_from_num_actions(num_actions: int) -> int:
    # For Go in OpenSpiel the action space is N*N + 1 (the +1 is pass).
    n2_plus_pass = num_actions
    n2 = n2_plus_pass - 1
    n = int(round(math.sqrt(n2)))
    if n * n + 1 != num_actions:
        raise ValueError(f"Unexpected action count {num_actions}; expected N*N+1.")
    return n


def a_to_rc(a: int, N: int) -> Tuple[int, int]:
    return (a // N, a % N)


def rc_to_a(r: int, c: int, N: int) -> int:
    return r * N + c


def pass_action(N: int) -> int:
    return N * N

