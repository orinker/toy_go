import numpy as np


def load_go_game(board_size=9, komi=7.5):
    # You can also call: pyspiel.load_game("go", {"board_size": board_size, "komi": komi})
    # Import locally to avoid hard dependency at module import time.
    import pyspiel  # OpenSpiel Python API

    return pyspiel.load_game(f"go(board_size={board_size},komi={komi})")


def obs_to_planes(game, state) -> np.ndarray:
    """
    Convert OpenSpiel observation tensor to (C,H,W) float32 for CNN input.
    """
    obs = np.array(state.observation_tensor(), dtype=np.float32)
    shape = game.observation_tensor_shape()
    if len(shape) == 3:
        c, h, w = shape
        planes = obs.reshape(c, h, w)
        return planes
    elif len(shape) == 1:
        raise RuntimeError(
            f"Got flat obs of length {shape[0]}; please reshape per your build."
        )
    else:
        raise RuntimeError(f"Unexpected observation shape: {shape}")
