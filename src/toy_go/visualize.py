
import math
from pathlib import Path

import pygame
import torch

from .game import load_go_game
from .mcts import MCTS
from .net import AZNet
from .utils import a_to_rc, pass_action


class GoVisualizer:
    """Pygame-based visualizer for Go boards."""

    BOARD_COLOR = (220, 179, 92)
    GRID_COLOR = (0, 0, 0)
    BLACK_STONE = (20, 20, 20)
    WHITE_STONE = (240, 240, 240)
    BACKGROUND = (245, 235, 220)
    TEXT_COLOR = (40, 40, 40)
    LAST_MOVE_MARK = (255, 0, 0)

    def __init__(
        self,
        net: AZNet,
        game,
        board_size: int,
        device: str = "cpu",
        mcts_sims: int = 100,
        cell_size: int = 60,
        board_margin: int = 40,
        stone_radius: int = 25,
        komi: float = 2.5,
    ):
        self.net = net
        self.game = game
        self.board_size = board_size
        self.cell_size = cell_size
        self.board_margin = board_margin
        self.stone_radius = stone_radius
        self.device = device
        self.mcts_sims = mcts_sims
        self.mcts = MCTS(game, net, device=device, dirichlet_eps=0.0)
        self.komi = komi

        pygame.init()
        self.width = self.board_size * self.cell_size + 2 * self.board_margin + 300
        self.height = self.board_size * self.cell_size + 2 * self.board_margin
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"{self.board_size}x{self.board_size} Go - AlphaZero")

        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)
        self.clock = pygame.time.Clock()

        self.state = game.new_initial_state()
        self.move_history = []
        self.paused = False
        self.game_over = False
        self.auto_play_delay = 1000
        self.last_move_time = 0
        self.last_win_rates: tuple[float, float] | None = None
        self.temp_moves = 10
        self.controls_help: list[str] = [
            "Controls:",
            "SPACE - Pause/Resume",
            "R - Reset game",
            "Q - Quit",
            "↑/↓ - Adjust speed",
        ]
        self.show_delay = True

    def grid_to_pixel(self, row: int, col: int) -> tuple[int, int]:
        """Convert board coordinates to pixel coordinates."""
        x = self.board_margin + col * self.cell_size + self.cell_size // 2
        y = self.board_margin + row * self.cell_size + self.cell_size // 2
        return x, y

    def draw_board(self):
        """Draw the Go board grid."""
        self.screen.fill(self.BACKGROUND)

        board_rect = pygame.Rect(
            self.board_margin,
            self.board_margin,
            self.board_size * self.cell_size,
            self.board_size * self.cell_size,
        )
        pygame.draw.rect(self.screen, self.BOARD_COLOR, board_rect)

        for i in range(self.board_size):
            start_x = self.board_margin + self.cell_size // 2
            start_y = self.board_margin + i * self.cell_size + self.cell_size // 2
            end_x = (
                self.board_margin
                + (self.board_size - 1) * self.cell_size
                + self.cell_size // 2
            )
            pygame.draw.line(self.screen, self.GRID_COLOR, (start_x, start_y), (end_x, start_y), 1)

            start_y = self.board_margin + self.cell_size // 2
            start_x = self.board_margin + i * self.cell_size + self.cell_size // 2
            end_y = (
                self.board_margin
                + (self.board_size - 1) * self.cell_size
                + self.cell_size // 2
            )
            pygame.draw.line(self.screen, self.GRID_COLOR, (start_x, start_y), (start_x, end_y), 1)

        for row, col in self._star_points():
            x, y = self.grid_to_pixel(row, col)
            pygame.draw.circle(self.screen, self.GRID_COLOR, (x, y), 4)

    def _star_points(self) -> list[tuple[int, int]]:
        """Star point coordinates for the current board size."""
        if self.board_size < 5:
            return []
        center = self.board_size // 2
        if self.board_size >= 9:
            offsets = {2, center, self.board_size - 3}
        elif self.board_size >= 7:
            offsets = {1, center, self.board_size - 2}
        else:  # 5x5
            offsets = {center}
        positions = sorted(offsets)
        return [(r, c) for r in positions for c in positions]

    def draw_stones(self):
        """Draw stones on the board based on current state."""
        for row, line in enumerate(self._board_lines()):
            for col, char in enumerate(line):
                if char == 'X':
                    x, y = self.grid_to_pixel(row, col)
                    pygame.draw.circle(self.screen, self.BLACK_STONE, (x, y), self.stone_radius)
                    pygame.draw.circle(self.screen, self.GRID_COLOR, (x, y), self.stone_radius, 1)
                elif char == 'O':
                    x, y = self.grid_to_pixel(row, col)
                    pygame.draw.circle(self.screen, self.WHITE_STONE, (x, y), self.stone_radius)
                    pygame.draw.circle(self.screen, self.GRID_COLOR, (x, y), self.stone_radius, 1)

        if self.move_history:
            last_action = self.move_history[-1]
            if last_action != pass_action(self.board_size):
                action_row, action_col = a_to_rc(last_action, self.board_size)
                # OpenSpiel: row 0 = bottom (Go row 1); display uses row 0 at top.
                display_row = self.board_size - 1 - action_row
                x, y = self.grid_to_pixel(display_row, action_col)
                pygame.draw.circle(self.screen, self.LAST_MOVE_MARK, (x, y), 5)

    def _board_lines(self) -> list[str]:
        """Extract board rows from the OpenSpiel state string."""
        rows: list[str] = []
        for line in self.state.to_string().splitlines():
            chars = [ch for ch in line if ch in "+OX"]
            if len(chars) == self.board_size:
                rows.append("".join(chars))
        return rows

    def draw_info(self):
        """Draw game information panel."""
        info_x = self.board_margin + self.board_size * self.cell_size + 20

        player = "Black" if self.state.current_player() == 0 else "White"
        move_text = self.font.render(f"Move: {len(self.move_history)}", True, self.TEXT_COLOR)
        self.screen.blit(move_text, (info_x, 40))

        player_text = self.font.render(f"To play: {player}", True, self.TEXT_COLOR)
        self.screen.blit(player_text, (info_x, 70))

        if self.last_win_rates is not None:
            black_wr, white_wr = self.last_win_rates
            black_wr_text = self.small_font.render(
                f"Black win%: {black_wr * 100:.1f}", True, self.TEXT_COLOR
            )
            white_wr_text = self.small_font.render(
                f"White win%: {white_wr * 100:.1f}", True, self.TEXT_COLOR
            )
            self.screen.blit(black_wr_text, (info_x, 100))
            self.screen.blit(white_wr_text, (info_x, 125))
            y_offset = 160
        else:
            y_offset = 100

        if self.game_over:
            returns = self.state.returns()
            black_score = returns[0]
            white_score = returns[1]
            raw_black, raw_white = self._final_raw_scores()

            # Determine winner
            if black_score > 0:
                winner = "Black"
                winner_color = self.BLACK_STONE
            else:
                winner = "White"
                winner_color = self.WHITE_STONE

            # Display winner with larger font
            result_text = self.font.render(f"Winner: {winner}", True, winner_color)
            self.screen.blit(result_text, (info_x, y_offset))
            y_offset += 35

            # Display individual scores
            black_score_text = self.font.render(
                f"Black (with komi): {black_score:.1f}", True, self.TEXT_COLOR
            )
            self.screen.blit(black_score_text, (info_x, y_offset))
            y_offset += 30

            white_score_text = self.font.render(
                f"White (with komi): {white_score:.1f}", True, self.TEXT_COLOR
            )
            self.screen.blit(white_score_text, (info_x, y_offset))
            y_offset += 30

            raw_black_text = self.small_font.render(
                f"Black raw: {raw_black:.1f}", True, self.TEXT_COLOR
            )
            self.screen.blit(raw_black_text, (info_x, y_offset))
            y_offset += 25

            raw_white_text = self.small_font.render(
                f"White raw: {raw_white:.1f}", True, self.TEXT_COLOR
            )
            self.screen.blit(raw_white_text, (info_x, y_offset))
            y_offset += 25

            # Display score difference
            diff = abs(black_score)
            diff_text = self.small_font.render(
                f"Margin (with komi): {diff:.1f}", True, self.TEXT_COLOR
            )
            self.screen.blit(diff_text, (info_x, y_offset))
            y_offset += 30

        if y_offset < 220:
            y_offset = 220
        for text in self.controls_help:
            control_text = self.small_font.render(text, True, self.TEXT_COLOR)
            self.screen.blit(control_text, (info_x, y_offset))
            y_offset += 25

        status_text = self.font.render(self._status_text(), True, self.TEXT_COLOR)
        self.screen.blit(status_text, (info_x, y_offset + 20))

        if self.show_delay:
            speed_text = self.small_font.render(
                f"Delay: {self.auto_play_delay}ms", True, self.TEXT_COLOR
            )
            self.screen.blit(speed_text, (info_x, y_offset + 50))

    def _status_text(self) -> str:
        if self.game_over:
            return "GAME OVER"
        return "PAUSED" if self.paused else "PLAYING"

    def _compute_win_rates(self, root) -> tuple[float, float]:
        if root is None or root.N == 0 or not math.isfinite(root.Q):
            return 0.5, 0.5
        value = float(max(min(root.Q, 1.0), -1.0))
        prob_current = 0.5 * (value + 1.0)
        prob_current = min(max(prob_current, 0.0), 1.0)
        if root.to_play == 0:
            return prob_current, 1.0 - prob_current
        return 1.0 - prob_current, prob_current

    def _final_raw_scores(self) -> tuple[float, float]:
        returns = self.state.returns()
        raw_scores: tuple[float, float] | None = None

        score_fn = getattr(self.state, "score", None)
        if callable(score_fn):
            try:
                raw_scores = (float(score_fn(0)), float(score_fn(1)))
            except TypeError:
                try:
                    raw_value = float(score_fn())
                    raw_scores = (raw_value, -raw_value)
                except Exception:
                    raw_scores = None

        if raw_scores is None:
            scores_fn = getattr(self.state, "scores", None)
            if callable(scores_fn):
                try:
                    values = scores_fn()
                    if len(values) >= 2:
                        raw_scores = (float(values[0]), float(values[1]))
                except Exception:
                    raw_scores = None

        if raw_scores is None:
            raw_diff = float(returns[0])
            if getattr(self, "komi", None):
                raw_diff += float(self.komi)
            raw_scores = (raw_diff, -raw_diff)

        return raw_scores

    def make_ai_move(self):
        """Make an AI move using MCTS."""
        if self.state.is_terminal():
            self.game_over = True
            return

        tau = 1.0 if len(self.move_history) < self.temp_moves else 1e-6
        _, action, root = self.mcts.run(
            self.state,
            num_sims=self.mcts_sims,
            temperature=tau,
        )
        self.last_win_rates = self._compute_win_rates(root)

        self.move_history.append(action)
        self.state.apply_action(action)
        self.mcts.advance(action)

        if self.state.is_terminal():
            self.game_over = True

    def reset_game(self):
        """Reset the game to initial state."""
        self.state = self.game.new_initial_state()
        self.move_history = []
        self.game_over = False
        self.mcts = MCTS(self.game, self.net, device=self.device, dirichlet_eps=0.0)
        self.last_win_rates = None

    def run(self):
        """Main game loop."""
        running = True

        while running:
            current_time = pygame.time.get_ticks()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_r:
                        self.reset_game()
                    elif event.key == pygame.K_UP:
                        self.auto_play_delay = min(5000, self.auto_play_delay + 100)
                    elif event.key == pygame.K_DOWN:
                        self.auto_play_delay = max(100, self.auto_play_delay - 100)

            if not self.paused and not self.game_over:
                if current_time - self.last_move_time > self.auto_play_delay:
                    self.make_ai_move()
                    self.last_move_time = current_time

            self.draw_board()
            self.draw_stones()
            self.draw_info()

            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()


def main_visualize(args):
    """Main entry point for visualization."""
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

    visualizer = GoVisualizer(
        net,
        game,
        board_size,
        device,
        args.mcts_sims,
        komi=komi,
    )
    visualizer.run()
