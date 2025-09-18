from __future__ import annotations

from pathlib import Path

import pygame
import torch

from .game import load_go_game
from .net import AZNet
from .utils import pass_action, rc_to_a
from .visualize import GoVisualizer


class GoPvCVisualizer(GoVisualizer):
    """Human vs AI Go interface built on top of the visualizer."""

    def __init__(
        self,
        net: AZNet,
        game,
        board_size: int,
        device: str = "cpu",
        mcts_sims: int = 200,
        komi: float = 2.5,
        temp_moves: int = 10,
    ) -> None:
        super().__init__(
            net,
            game,
            board_size,
            device=device,
            mcts_sims=mcts_sims,
            komi=komi,
        )
        self.human_player = 0
        self.ai_player = 1
        self.temp_moves = temp_moves
        self.controls_help = [
            "Controls:",
            "Left click - Play stone",
            "P - Pass",
            "R - Reset game",
            "Q - Quit",
        ]
        self.show_delay = False

    def _status_text(self) -> str:
        if self.game_over:
            return "GAME OVER"
        if self.state.current_player() == self.human_player:
            return "Your move"
        return "AI thinking"

    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_keydown(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)

            if not self.game_over and self.state.current_player() == self.ai_player:
                self.make_ai_move()

            self.draw_board()
            self.draw_stones()
            self.draw_info()
            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()

    def _handle_keydown(self, key: int) -> bool:
        if key == pygame.K_q:
            return False
        if key == pygame.K_r:
            self.reset_game()
            return True
        if key == pygame.K_p:
            self._handle_pass()
            return True
        return True

    def _handle_click(self, pos: tuple[int, int]) -> None:
        if self.game_over or self.state.current_player() != self.human_player:
            return

        coords = self._pixel_to_board(pos)
        if coords is None:
            return

        row, col = coords
        action = rc_to_a(row, col, self.board_size)
        legal = self.state.legal_actions()
        if action not in legal:
            return
        self._apply_human_action(action)

    def _handle_pass(self) -> None:
        if self.game_over or self.state.current_player() != self.human_player:
            return
        action = pass_action(self.board_size)
        if action not in self.state.legal_actions():
            return
        self._apply_human_action(action)

    def _apply_human_action(self, action: int) -> None:
        self.move_history.append(action)
        self.state.apply_action(action)
        self.mcts.advance(action)
        if self.state.is_terminal():
            self.game_over = True

    def _pixel_to_board(self, pos: tuple[int, int]) -> tuple[int, int] | None:
        x, y = pos
        board_left = self.board_margin
        board_top = self.board_margin
        board_right = board_left + self.board_size * self.cell_size
        board_bottom = board_top + self.board_size * self.cell_size
        if not (board_left <= x <= board_right and board_top <= y <= board_bottom):
            return None

        origin_x = board_left + 0.5 * self.cell_size
        origin_y = board_top + 0.5 * self.cell_size
        col_f = (x - origin_x) / self.cell_size
        row_f = (y - origin_y) / self.cell_size
        col = int(round(col_f))
        row = int(round(row_f))
        if not (0 <= col < self.board_size and 0 <= row < self.board_size):
            return None
        if abs(col_f - col) > 0.4 or abs(row_f - row) > 0.4:
            return None

        go_row = self.board_size - 1 - row
        return go_row, col


def main_pvc(args) -> None:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    board_size = args.board_size
    komi = args.komi
    game = load_go_game(board_size=board_size, komi=komi)
    C, H, W = game.observation_tensor_shape()
    if H != board_size or W != board_size:
        raise ValueError(
            f"Observation shape mismatch: expected {board_size}x{board_size}, got {H}x{W}"
        )

    net = AZNet(
        in_planes=C, board_size=board_size, channels=args.channels, blocks=args.blocks
    ).to(device)
    if args.ckpt:
        ckpt_path = Path(args.ckpt).expanduser()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location=device)
        if isinstance(payload, dict) and "model" in payload:
            net.load_state_dict(payload["model"])
        else:
            net.load_state_dict(payload)
    net.eval()

    ui = GoPvCVisualizer(
        net,
        game,
        board_size,
        device=device,
        mcts_sims=args.mcts_sims,
        komi=komi,
    )
    ui.run()
