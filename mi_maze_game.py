"""Cartoon maze mini-game driven by the MI real-time classifier."""
from __future__ import annotations

import argparse
import random
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import numpy as np
import pygame
from pylsl import StreamInlet, resolve_byprop

from algorithms.fbcsp import FilterBankCSPClassifier
from config import mi_config as cfg


LABEL_TO_COMMAND = {
    0: cfg.IDLE_COMMAND,
    1: "UP",
    2: "DOWN",
    3: "LEFT",
    4: "RIGHT",
}

COMMAND_TO_OFFSET = {
    cfg.IDLE_COMMAND: (0, 0),
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Maze runner controlled by MI predictions")
    parser.add_argument("model_dir", help="Directory containing the trained model artifacts")
    parser.add_argument("--stream-name", default=cfg.LSL_STREAM_NAME, help="LSL stream name to resolve")
    parser.add_argument("--maze-cols", type=int, default=11, help="Number of maze columns (odd preferred)")
    parser.add_argument("--maze-rows", type=int, default=11, help="Number of maze rows (odd preferred)")
    parser.add_argument(
        "--move-interval",
        type=float,
        default=0.65,
        help="Seconds between consecutive moves while holding the same command",
    )
    parser.add_argument(
        "--maze-open-factor",
        type=float,
        default=0.35,
        help="Fraction (0-1) of extra walls to remove for easier mazes",
    )
    parser.add_argument(
        "--single-step",
        dest="single_step",
        action="store_true",
        default=True,
        help="Require the decoded command to change before moving again (default)",
    )
    parser.add_argument(
        "--continuous-step",
        dest="single_step",
        action="store_false",
        help="Allow sustained commands to move multiple cells without pausing",
    )
    return parser.parse_args()


def connect_lsl(stream_name: str) -> StreamInlet:
    streams = resolve_byprop("name", stream_name, timeout=10)
    if not streams:
        raise RuntimeError(f"Could not find LSL stream named {stream_name}")
    return StreamInlet(streams[0], max_buflen=60)


def load_sprite(cell_size: int) -> Optional[pygame.Surface]:
    sprite_path = Path("res") / "xiaohui.png"
    if not sprite_path.exists():
        print(f"Warning: sprite not found at {sprite_path}, fallback circle will be used.")
        return None
    sprite = pygame.image.load(str(sprite_path)).convert_alpha()
    scale = max(1, int(cell_size * 0.8))
    return pygame.transform.smoothscale(sprite, (scale, scale))


def init_display() -> Tuple[pygame.Surface, pygame.time.Clock]:
    pygame.init()
    pygame.display.set_caption("MI Maze Runner")
    screen = pygame.display.set_mode(cfg.SCREEN_SIZE)
    return screen, pygame.time.Clock()


def load_unicode_font(size: int) -> pygame.font.Font:
    """Return a font that can render ASCII + CJK text when available."""
    candidates = [
        Path("res") / "fonts" / "NotoSansSC-Regular.otf",
        Path("res") / "fonts" / "NotoSansSC-Regular.ttf",
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyh.ttf"),
        Path("C:/Windows/Fonts/msjh.ttc"),
        Path("C:/Windows/Fonts/msjh.ttf"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simhei.ttc"),
    ]
    for path in candidates:
        try:
            if path.exists():
                return pygame.font.Font(str(path), size)
        except Exception:
            continue
    fallback = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", cfg.FONT_NAME]
    for name in fallback:
        try:
            return pygame.font.SysFont(name, size)
        except Exception:
            continue
    return pygame.font.Font(None, size)


def draw_overlay(screen: pygame.Surface, title: str, subtitle: str, font_main: pygame.font.Font, font_sub: pygame.font.Font) -> None:
    overlay = pygame.Surface(cfg.SCREEN_SIZE, pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))
    title_surf = font_main.render(title, True, (255, 255, 255))
    title_rect = title_surf.get_rect(center=(cfg.SCREEN_SIZE[0] // 2, cfg.SCREEN_SIZE[1] // 2 - 30))
    overlay.blit(title_surf, title_rect)
    sub_surf = font_sub.render(subtitle, True, (220, 220, 230))
    sub_rect = sub_surf.get_rect(center=(cfg.SCREEN_SIZE[0] // 2, title_rect.bottom + 40))
    overlay.blit(sub_surf, sub_rect)
    screen.blit(overlay, (0, 0))


def majority_vote(history: Deque[str]) -> str:
    counts = Counter(history)
    return counts.most_common(1)[0][0] if counts else cfg.IDLE_COMMAND


@dataclass
class CellWalls:
    N: bool = True
    S: bool = True
    E: bool = True
    W: bool = True


class Maze:
    def __init__(self, rows: int, cols: int, open_factor: float = 0.0) -> None:
        self.rows = max(3, rows)
        self.cols = max(3, cols)
        self.open_factor = max(0.0, min(1.0, open_factor))
        self.grid: List[List[CellWalls]] = [
            [CellWalls() for _ in range(self.cols)]
            for _ in range(self.rows)
        ]
        self.start = (self.rows // 2, self.cols // 2)
        self.exit = self.start
        self._carve_maze()
        self._open_start_cross()
        self._open_extra_passages()

    def _neighbors(self, row: int, col: int) -> List[Tuple[int, int, str]]:
        refs = [(-1, 0, "N"), (1, 0, "S"), (0, 1, "E"), (0, -1, "W")]
        res = []
        for dr, dc, label in refs:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                res.append((nr, nc, label))
        return res

    def _remove_wall(self, a: Tuple[int, int], b: Tuple[int, int], direction: str) -> None:
        opposite = {"N": "S", "S": "N", "E": "W", "W": "E"}
        ar, ac = a
        br, bc = b
        setattr(self.grid[ar][ac], direction, False)
        setattr(self.grid[br][bc], opposite[direction], False)

    def _carve_maze(self) -> None:
        stack = [self.start]
        visited = {self.start}
        while stack:
            r, c = stack[-1]
            candidates = [n for n in self._neighbors(r, c) if (n[0], n[1]) not in visited]
            if candidates:
                nr, nc, direction = random.choice(candidates)
                self._remove_wall((r, c), (nr, nc), direction)
                visited.add((nr, nc))
                stack.append((nr, nc))
            else:
                stack.pop()
        boundary = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if r in {0, self.rows - 1} or c in {0, self.cols - 1}
        ]
        boundary = [cell for cell in boundary if cell != self.start]
        self.exit = random.choice(boundary) if boundary else self.start

    def can_move(self, position: Tuple[int, int], command: str) -> bool:
        row, col = position
        cell = self.grid[row][col]
        if command == "UP":
            return not cell.N
        if command == "DOWN":
            return not cell.S
        if command == "LEFT":
            return not cell.W
        if command == "RIGHT":
            return not cell.E
        return False

    def _open_start_cross(self) -> None:
        r, c = self.start
        for nr, nc, direction in self._neighbors(r, c):
            self._remove_wall((r, c), (nr, nc), direction)

    def _open_extra_passages(self) -> None:
        if self.open_factor <= 0.0:
            return
        attempts = int(self.rows * self.cols * self.open_factor)
        for _ in range(attempts):
            r = random.randrange(self.rows)
            c = random.randrange(self.cols)
            neighbors = self._neighbors(r, c)
            if not neighbors:
                continue
            nr, nc, direction = random.choice(neighbors)
            cell = self.grid[r][c]
            if getattr(cell, direction):
                self._remove_wall((r, c), (nr, nc), direction)


class MazeGame:
    def __init__(
        self,
        rows: int,
        cols: int,
        move_interval: float,
        open_factor: float,
        single_step_mode: bool,
    ) -> None:
        rows = rows if rows % 2 == 1 else rows + 1
        cols = cols if cols % 2 == 1 else cols + 1
        self.maze = Maze(rows, cols, open_factor)
        self.move_interval = max(0.2, move_interval)
        self.player_cell = self.maze.start
        self.last_move_ts = 0.0
        self.last_step_direction: Optional[str] = None
        self.single_step_mode = single_step_mode
        self.start_ts = time.perf_counter()
        self.finished_at: Optional[float] = None
        self.elapsed = 0.0
        self.font_big = pygame.font.SysFont(cfg.FONT_NAME, 48)
        self.font_small = pygame.font.SysFont(cfg.FONT_NAME, 26)
        self.cell_size = self._compute_cell_size()
        self.offset = (
            (cfg.SCREEN_SIZE[0] - self.maze.cols * self.cell_size) // 2,
            (cfg.SCREEN_SIZE[1] - self.maze.rows * self.cell_size) // 2,
        )
        exit_font_px = max(14, min(40, int(self.cell_size * 0.55)))
        self.font_exit = pygame.font.SysFont(cfg.FONT_NAME, exit_font_px)
        self.sprite = load_sprite(self.cell_size)

    def _compute_cell_size(self) -> int:
        margin = 120
        size = min(
            (cfg.SCREEN_SIZE[0] - margin) // self.maze.cols,
            (cfg.SCREEN_SIZE[1] - margin) // self.maze.rows,
        )
        return max(16, size)

    def update_timer(self) -> None:
        if self.finished_at is None:
            self.elapsed = time.perf_counter() - self.start_ts
        else:
            self.elapsed = self.finished_at - self.start_ts

    def try_move(self, command: str) -> None:
        if self.finished_at is not None:
            return
        if command == cfg.IDLE_COMMAND:
            if self.single_step_mode:
                self.last_step_direction = None
            return
        now = time.perf_counter()
        if now - self.last_move_ts < self.move_interval:
            return
        if not self.maze.can_move(self.player_cell, command):
            return
        if self.single_step_mode and self.last_step_direction == command:
            return
        dr, dc = COMMAND_TO_OFFSET[command]
        row, col = self.player_cell
        self.player_cell = (row + dr, col + dc)
        self.last_move_ts = now
        if self.single_step_mode:
            self.last_step_direction = command
        if self.player_cell == self.maze.exit:
            self.finished_at = now

    def draw(self, screen: pygame.Surface, command: str, confidence: float) -> None:
        screen.fill((12, 12, 28))
        ox, oy = self.offset
        cs = self.cell_size
        base_color = (50, 70, 110)
        for r in range(self.maze.rows):
            for c in range(self.maze.cols):
                rect = pygame.Rect(ox + c * cs, oy + r * cs, cs, cs)
                pygame.draw.rect(screen, base_color, rect)
        wall_color = (15, 20, 35)
        for r in range(self.maze.rows):
            for c in range(self.maze.cols):
                cell = self.maze.grid[r][c]
                x, y = ox + c * cs, oy + r * cs
                if cell.N:
                    pygame.draw.line(screen, wall_color, (x, y), (x + cs, y), 3)
                if cell.S:
                    pygame.draw.line(screen, wall_color, (x, y + cs), (x + cs, y + cs), 3)
                if cell.W:
                    pygame.draw.line(screen, wall_color, (x, y), (x, y + cs), 3)
                if cell.E:
                    pygame.draw.line(screen, wall_color, (x + cs, y), (x + cs, y + cs), 3)
        exit_margin = max(3, int(cs * 0.2))
        exit_rect = pygame.Rect(
            ox + self.maze.exit[1] * cs + exit_margin,
            oy + self.maze.exit[0] * cs + exit_margin,
            cs - 2 * exit_margin,
            cs - 2 * exit_margin,
        )
        pygame.draw.rect(screen, (140, 200, 120), exit_rect)
        exit_label = self.font_exit.render("EXIT", True, (25, 60, 25))
        exit_label_rect = exit_label.get_rect(center=exit_rect.center)
        screen.blit(exit_label, exit_label_rect)
        px = ox + self.player_cell[1] * cs + cs // 2
        py = oy + self.player_cell[0] * cs + cs // 2
        if self.sprite is not None:
            sprite_rect = self.sprite.get_rect(center=(px, py))
            screen.blit(self.sprite, sprite_rect)
        else:
            pygame.draw.circle(screen, (240, 180, 90), (px, py), cs // 3)
        timer_text = self.font_big.render(f"Time: {self.elapsed:5.2f}s", True, (255, 255, 255))
        screen.blit(timer_text, (40, 30))
        command_text = self.font_small.render(
            f"Command: {command}  Conf: {confidence:.2f}", True, (220, 220, 240)
        )
        screen.blit(command_text, (40, 90))
        hint = self.font_small.render(
            "Guide Xiaohui to the green exit to escape the maze",
            True,
            (200, 200, 200),
        )
        screen.blit(hint, (40, cfg.SCREEN_SIZE[1] - 70))
        if self.finished_at is not None:
            banner = self.font_big.render("Great job!", True, (255, 215, 0))
            banner_rect = banner.get_rect(center=(cfg.SCREEN_SIZE[0] // 2, 60))
            screen.blit(banner, banner_rect)


def run_game(model: FilterBankCSPClassifier, inlet: StreamInlet, args: argparse.Namespace) -> None:
    screen, clock = init_display()
    buffer: List[List[float]] = []
    samples_since_inference = 0
    window_samples = int(cfg.SLIDING_WINDOW_SEC * cfg.SAMPLE_RATE_HZ)
    step_samples = int(cfg.WINDOW_STEP_SEC * cfg.SAMPLE_RATE_HZ)
    max_buffer_samples = window_samples * 3
    history: Deque[str] = deque(maxlen=cfg.MAJORITY_VOTE_WINDOW)
    current_command = cfg.IDLE_COMMAND
    current_conf = 0.0
    game: Optional[MazeGame] = None
    state = "waiting"  # waiting, running, finished
    overlay_font = load_unicode_font(56)
    overlay_small = load_unicode_font(28)

    def start_game() -> None:
        nonlocal game, state, current_command, current_conf, samples_since_inference
        game = MazeGame(
            args.maze_rows,
            args.maze_cols,
            args.move_interval,
            args.maze_open_factor,
            args.single_step,
        )
        buffer.clear()
        history.clear()
        samples_since_inference = 0
        current_command = cfg.IDLE_COMMAND
        current_conf = 0.0
        state = "running"

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise SystemExit
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    if state in {"waiting", "finished"}:
                        start_game()

            chunk, _ = inlet.pull_chunk(timeout=0.0, max_samples=int(cfg.SAMPLE_RATE_HZ * cfg.CHUNK_LENGTH_SEC))
            if chunk:
                buffer.extend(sample[: cfg.EXPECTED_CHANNEL_COUNT] for sample in chunk)
                samples_since_inference += len(chunk)
                if len(buffer) > max_buffer_samples:
                    buffer = buffer[-max_buffer_samples:]

            if state != "running" or game is None:
                if state == "waiting":
                    screen.fill((12, 12, 28))
                    draw_overlay(screen, "Press Space to Begin", "Ensure the EEG stream is live", overlay_font, overlay_small)
                elif state == "finished" and game is not None:
                    game.draw(screen, current_command, current_conf)
                    draw_overlay(screen, "Great job!", "Press Space to play again", overlay_font, overlay_small)
                pygame.display.flip()
                clock.tick(cfg.REFRESH_RATE_HZ)
                continue

            if len(buffer) >= window_samples and samples_since_inference >= step_samples:
                window = np.asarray(buffer[-window_samples:])
                trial = np.expand_dims(window.T, axis=0)
                probs = model.predict_proba(trial)[0]
                best_idx = int(np.argmax(probs))
                best_label = int(model.classes_[best_idx])
                best_conf = float(probs[best_idx])
                command = LABEL_TO_COMMAND.get(best_label, cfg.IDLE_COMMAND)
                history.append(command if best_conf >= cfg.CONFIDENCE_THRESHOLD else cfg.IDLE_COMMAND)
                current_command = majority_vote(history)
                current_conf = best_conf
                samples_since_inference = 0

            game.try_move(current_command)
            game.update_timer()
            if game.finished_at is not None and state == "running":
                state = "finished"

            game.draw(screen, current_command, current_conf)
            if state == "finished":
                draw_overlay(screen, "Great job!", "Press Space to play again", overlay_font, overlay_small)
            pygame.display.flip()
            clock.tick(cfg.REFRESH_RATE_HZ)
    finally:
        pygame.quit()


def main() -> None:
    args = parse_args()
    model = FilterBankCSPClassifier.load(Path(args.model_dir))
    inlet = connect_lsl(args.stream_name)
    run_game(model, inlet, args)


if __name__ == "__main__":
    main()
