import numpy as np

from main import infer_board_size_from_num_actions, a_to_rc, rc_to_a, pass_action


def test_infer_board_size_from_num_actions():
    assert infer_board_size_from_num_actions(82) == 9  # 9*9 + 1
    assert infer_board_size_from_num_actions(26) == 5  # 5*5 + 1


def test_rc_a_roundtrip():
    N = 9
    for r in [0, 4, 8]:
        for c in [0, 4, 8]:
            a = rc_to_a(r, c, N)
            rr, cc = a_to_rc(a, N)
            assert (rr, cc) == (r, c)


def test_pass_action_index():
    assert pass_action(9) == 81

