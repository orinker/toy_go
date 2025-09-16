import pygame
import torch
from dataclasses import dataclass
from typing import Optional, Tuple

from .game import load_go_game
from .mcts import MCTS
from .net import AZNet
from .utils import a_to_rc, pass_action


@dataclass
class GoVisualizer:
    """Pygame-based visualizer for 9x9 Go game."""

    BOARD_SIZE: int = 9
    CELL_SIZE: int = 60
    BOARD_MARGIN: int = 40
    STONE_RADIUS: int = 25

    BOARD_COLOR = (220, 179, 92)
    GRID_COLOR = (0, 0, 0)
    BLACK_STONE = (20, 20, 20)
    WHITE_STONE = (240, 240, 240)
    BACKGROUND = (245, 235, 220)
    TEXT_COLOR = (40, 40, 40)
    LAST_MOVE_MARK = (255, 0, 0)

    def __init__(self, net: AZNet, game, device: str = "cpu", mcts_sims: int = 100):
        self.net = net
        self.game = game
        self.device = device
        self.mcts_sims = mcts_sims
        self.mcts = MCTS(game, net, device=device)

        pygame.init()
        self.width = self.BOARD_SIZE * self.CELL_SIZE + 2 * self.BOARD_MARGIN + 300
        self.height = self.BOARD_SIZE * self.CELL_SIZE + 2 * self.BOARD_MARGIN
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("9x9 Go - AlphaZero")

        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)
        self.clock = pygame.time.Clock()

        self.state = game.new_initial_state()
        self.move_history = []
        self.paused = False
        self.game_over = False
        self.auto_play_delay = 1000
        self.last_move_time = 0

    def grid_to_pixel(self, row: int, col: int) -> Tuple[int, int]:
        """Convert board coordinates to pixel coordinates."""
        x = self.BOARD_MARGIN + col * self.CELL_SIZE + self.CELL_SIZE // 2
        y = self.BOARD_MARGIN + row * self.CELL_SIZE + self.CELL_SIZE // 2
        return x, y

    def draw_board(self):
        """Draw the Go board grid."""
        self.screen.fill(self.BACKGROUND)

        board_rect = pygame.Rect(
            self.BOARD_MARGIN,
            self.BOARD_MARGIN,
            self.BOARD_SIZE * self.CELL_SIZE,
            self.BOARD_SIZE * self.CELL_SIZE
        )
        pygame.draw.rect(self.screen, self.BOARD_COLOR, board_rect)

        for i in range(self.BOARD_SIZE):
            start_x = self.BOARD_MARGIN + self.CELL_SIZE // 2
            start_y = self.BOARD_MARGIN + i * self.CELL_SIZE + self.CELL_SIZE // 2
            end_x = self.BOARD_MARGIN + (self.BOARD_SIZE - 1) * self.CELL_SIZE + self.CELL_SIZE // 2
            pygame.draw.line(self.screen, self.GRID_COLOR, (start_x, start_y), (end_x, start_y), 1)

            start_y = self.BOARD_MARGIN + self.CELL_SIZE // 2
            start_x = self.BOARD_MARGIN + i * self.CELL_SIZE + self.CELL_SIZE // 2
            end_y = self.BOARD_MARGIN + (self.BOARD_SIZE - 1) * self.CELL_SIZE + self.CELL_SIZE // 2
            pygame.draw.line(self.screen, self.GRID_COLOR, (start_x, start_y), (start_x, end_y), 1)

        star_points = [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)]
        for row, col in star_points:
            x, y = self.grid_to_pixel(row, col)
            pygame.draw.circle(self.screen, self.GRID_COLOR, (x, y), 4)

    def draw_stones(self):
        """Draw stones on the board based on current state."""
        board_str = self.state.to_string()
        lines = board_str.split('\n')[2:11]

        for row, line in enumerate(lines):
            # Find where the board positions start (after row number and space)
            # Row 9 has format "9 +++++++++", others have " 8 +++++++++"
            board_start = line.rfind(' ') + 1
            board_positions = line[board_start:board_start + 9]

            for col, char in enumerate(board_positions):
                if char == 'X':
                    x, y = self.grid_to_pixel(row, col)
                    pygame.draw.circle(self.screen, self.BLACK_STONE, (x, y), self.STONE_RADIUS)
                    pygame.draw.circle(self.screen, self.GRID_COLOR, (x, y), self.STONE_RADIUS, 1)
                elif char == 'O':
                    x, y = self.grid_to_pixel(row, col)
                    pygame.draw.circle(self.screen, self.WHITE_STONE, (x, y), self.STONE_RADIUS)
                    pygame.draw.circle(self.screen, self.GRID_COLOR, (x, y), self.STONE_RADIUS, 1)

        if self.move_history:
            last_action = self.move_history[-1]
            if last_action != pass_action(self.BOARD_SIZE):
                action_row, action_col = a_to_rc(last_action, self.BOARD_SIZE)
                # OpenSpiel: row 0 = bottom (Go row 1), row 8 = top (Go row 9)
                # Display: row 0 = top (Go row 9), row 8 = bottom (Go row 1)
                display_row = 8 - action_row
                x, y = self.grid_to_pixel(display_row, action_col)
                pygame.draw.circle(self.screen, self.LAST_MOVE_MARK, (x, y), 5)

    def draw_info(self):
        """Draw game information panel."""
        info_x = self.BOARD_MARGIN + self.BOARD_SIZE * self.CELL_SIZE + 20

        player = "Black" if self.state.current_player() == 0 else "White"
        move_text = self.font.render(f"Move: {len(self.move_history)}", True, self.TEXT_COLOR)
        self.screen.blit(move_text, (info_x, 40))

        player_text = self.font.render(f"To play: {player}", True, self.TEXT_COLOR)
        self.screen.blit(player_text, (info_x, 70))

        if self.game_over:
            returns = self.state.returns()
            winner = "Black" if returns[0] > 0 else "White"
            result_text = self.font.render(f"Winner: {winner}", True, self.TEXT_COLOR)
            self.screen.blit(result_text, (info_x, 100))
            score_text = self.small_font.render(f"Score: B{returns[0]:.1f}", True, self.TEXT_COLOR)
            self.screen.blit(score_text, (info_x, 130))

        y_offset = 200
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

        speed_text = self.small_font.render(f"Delay: {self.auto_play_delay}ms", True, self.TEXT_COLOR)
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

        if self.state.is_terminal():
            self.game_over = True

    def reset_game(self):
        """Reset the game to initial state."""
        self.state = self.game.new_initial_state()
        self.move_history = []
        self.game_over = False
        self.mcts = MCTS(self.game, self.net, device=self.device)

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
    game = load_go_game(board_size=9, komi=7.5)
    C, _, _ = game.observation_tensor_shape()

    net = AZNet(in_planes=C, board_size=9, channels=args.channels, blocks=args.blocks).to(device)
    if args.ckpt:
        net.load_state_dict(torch.load(args.ckpt, map_location=device))
    net.eval()

    visualizer = GoVisualizer(net, game, device, args.mcts_sims)
    visualizer.run()