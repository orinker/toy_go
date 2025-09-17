
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

        if self.game_over:
            returns = self.state.returns()
            black_score = returns[0]
            white_score = returns[1]

            # Determine winner
            if black_score > 0:
                winner = "Black"
                winner_color = self.BLACK_STONE
            else:
                winner = "White"
                winner_color = self.WHITE_STONE

            # Display winner with larger font
            result_text = self.font.render(f"Winner: {winner}", True, winner_color)
            self.screen.blit(result_text, (info_x, 100))

            # Display individual scores
            black_score_text = self.font.render(f"Black: {black_score:.1f}", True, self.TEXT_COLOR)
            self.screen.blit(black_score_text, (info_x, 130))

            white_score_text = self.font.render(f"White: {white_score:.1f}", True, self.TEXT_COLOR)
            self.screen.blit(white_score_text, (info_x, 155))

            # Display score difference
            diff = abs(black_score)
            diff_text = self.small_font.render(f"Margin: {diff:.1f} points", True, self.TEXT_COLOR)
            self.screen.blit(diff_text, (info_x, 180))

        y_offset = 220
        controls = [
            "Controls:",
            "SPACE - Pause/Resume",
            "R - Reset game",
            "Q - Quit",
            "↑/↓ - Adjust speed"
        ]
        for text in controls:
            control_text = self.small_font.render(text, True, self.TEXT_COLOR)
            self.screen.blit(control_text, (info_x, y_offset))
            y_offset += 25

        status = "PAUSED" if self.paused else "PLAYING"
        if self.game_over:
            status = "GAME OVER"
        status_text = self.font.render(status, True, self.TEXT_COLOR)
        self.screen.blit(status_text, (info_x, y_offset + 20))

        speed_text = self.small_font.render(
            f"Delay: {self.auto_play_delay}ms", True, self.TEXT_COLOR
        )
        self.screen.blit(speed_text, (info_x, y_offset + 50))

    def make_ai_move(self):
        """Make an AI move using MCTS."""
        if self.state.is_terminal():
            self.game_over = True
            return

        tau = 1.0 if len(self.move_history) < 10 else 1e-6
        _, action, _ = self.mcts.run(self.state, num_sims=self.mcts_sims, temperature=tau)

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

    visualizer = GoVisualizer(net, game, board_size, device, args.mcts_sims)
    visualizer.run()
