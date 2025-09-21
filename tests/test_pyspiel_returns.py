"""Tests describing the meaning of `pyspiel` Go returns."""

import pytest

from toy_go.utils import pass_action, rc_to_a

pyspiel = pytest.importorskip("pyspiel")


def test_go_returns_are_score_differences():
    """State.returns() exposes the final score difference, not a win flag."""

    board_size = 2
    komi = 0.0
    game = pyspiel.load_game(f"go(board_size={board_size},komi={komi})")
    go_pass = pass_action(board_size)

    # Empty board followed by a double-pass yields a zero score difference.
    state = game.new_initial_state()
    state.apply_action(go_pass)  # Black pass
    state.apply_action(go_pass)  # White pass
    assert state.is_terminal()
    assert state.returns() == [0.0, 0.0]

    # A single black stone left on the board scores +1 for black (area scoring).
    state = game.new_initial_state()
    state.apply_action(rc_to_a(0, 0, board_size))  # Black stone at (0, 0)
    state.apply_action(go_pass)  # White pass
    state.apply_action(go_pass)  # Black pass (second consecutive pass -> game end)
    assert state.is_terminal()
    returns = state.returns()
    assert returns[0] == pytest.approx(1.0)
    assert returns[1] == pytest.approx(-1.0)
    assert returns[0] == -returns[1]

def test_go_5x5_komi_0():
    """
    this create a game and see whether returns() returns the score difference
    """

    board_size = 5
    komi = 1.0
    game = pyspiel.load_game(f"go(board_size={board_size},komi={komi},scoring=area)")

    state = game.new_initial_state()
    
    black_cur = 0
    white_cur = 24
    go_pass = pass_action(board_size)
    
    while black_cur < 15:
        if black_cur < 15:
            state.apply_action(black_cur)
            if black_cur == 0 or black_cur == 2:
                black_cur += 2
            else:
                black_cur += 1
        else:
            state.apply_action(go_pass)
        
        if white_cur > 14:
            state.apply_action(white_cur)
            if white_cur == 24 or white_cur == 22:
                white_cur -= 2
            else:
                white_cur -= 1
        else:
            state.apply_action(go_pass)
    
    print(state)
    state.apply_action(go_pass)
    print(state.returns())

if __name__ == "__main__":
    test_go_5x5_komi_0()