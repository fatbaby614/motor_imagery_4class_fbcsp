"""Motor imagery controlled Tetris clone using Filter Bank CSP predictions."""
from __future__ import annotations

import argparse
import random
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pygame
from pylsl import StreamInlet, resolve_byprop

from algorithms.fbcsp import FilterBankCSPClassifier
from config import mi_config as cfg

LABEL_TO_ACTION = {
    0: "IDLE",
    1: "ROTATE",
    2: "SOFT_DROP",
    3: "MOVE_LEFT",
    4: "MOVE_RIGHT",
}

ACTION_DISPLAY = {
    "IDLE": "Idle",
    "ROTATE": "Rotate (tongue)",
    "SOFT_DROP": "Soft drop (foot)",
    "MOVE_LEFT": "Left (left hand)",
    "MOVE_RIGHT": "Right (right hand)",
}

ACTION_COOLDOWNS = {
    "MOVE_LEFT": 0.32,
    "MOVE_RIGHT": 0.32,
    "ROTATE": 0.45,
}

HORIZONTAL_INITIAL_DELAY = 0.35
HORIZONTAL_REPEAT_DELAY = 0.22
ROTATE_REPEAT_DELAY = ACTION_COOLDOWNS["ROTATE"]

LINE_SCORES = {1: 100, 2: 300, 3: 500, 4: 800}

TETROMINO_SHAPES: Dict[str, List[List[Tuple[int, int]]]] = {
    "I": [
        [(-1, 0), (0, 0), (1, 0), (2, 0)],
        [(0, -1), (0, 0), (0, 1), (0, 2)],
    ],
    "O": [
        [(0, 0), (1, 0), (0, 1), (1, 1)],
    ],
    "T": [
        [(-1, 0), (0, 0), (1, 0), (0, 1)],
        [(0, -1), (0, 0), (0, 1), (1, 0)],
        [(-1, 0), (0, 0), (1, 0), (0, -1)],
        [(0, -1), (0, 0), (0, 1), (-1, 0)],
    ],
    "L": [
        [(-1, 0), (0, 0), (1, 0), (1, 1)],
        [(0, -1), (0, 0), (0, 1), (1, -1)],
        [(-1, -1), (-1, 0), (0, 0), (1, 0)],
        [(-1, 1), (0, -1), (0, 0), (0, 1)],
    ],
    "J": [
        [(-1, 0), (0, 0), (1, 0), (-1, 1)],
        [(0, -1), (0, 0), (0, 1), (1, 1)],
        [(-1, 0), (0, 0), (1, 0), (1, -1)],
        [(-1, -1), (0, -1), (0, 0), (0, 1)],
    ],
    "S": [
        [(0, 0), (1, 0), (-1, 1), (0, 1)],
        [(0, -1), (0, 0), (1, 0), (1, 1)],
    ],
    "Z": [
        [(-1, 0), (0, 0), (0, 1), (1, 1)],
        [(1, -1), (0, 0), (1, 0), (0, 1)],
    ],
}

SHAPE_COLORS = {
    "I": (120, 220, 255),
    "O": (255, 230, 120),
    "T": (220, 160, 255),
    "L": (255, 190, 120),
    "J": (140, 200, 255),
    "S": (170, 245, 160),
    "Z": (255, 140, 140),
}


@dataclass
class Tetromino:
    name: str
    rotation: int
    x: int
    y: int

    def cells(self, rotation_override: Optional[int] = None, offset: Tuple[int, int] = (0, 0)) -> List[Tuple[int, int]]:
        rotation = rotation_override if rotation_override is not None else self.rotation
        pattern = TETROMINO_SHAPES[self.name][rotation % len(TETROMINO_SHAPES[self.name])]
        ox, oy = offset
        return [(self.x + dx + ox, self.y + dy + oy) for dx, dy in pattern]


class TetrisGame:
    def __init__(self, drop_interval: float = 0.8, soft_drop_scale: float = 0.25) -> None:
        self.width = 10
        self.height = 20
        self.drop_interval = max(0.2, drop_interval)
        self.soft_drop_scale = max(0.05, min(1.0, soft_drop_scale))
        usable_height = cfg.SCREEN_SIZE[1] - 160
        usable_width = int(cfg.SCREEN_SIZE[0] * 0.55)
        self.cell_size = max(18, min(38, min(usable_width // self.width, usable_height // self.height)))
        self.board_origin = (
            cfg.SCREEN_SIZE[0] // 2 - (self.width * self.cell_size) // 2,
            40,
        )
        self.rng = random.Random()
        self.reset()

    def reset(self) -> None:
        self.grid: List[List[Optional[str]]] = [[None for _ in range(self.width)] for _ in range(self.height)]
        self.score = 0
        self.lines = 0
        self.level = 1
        self.game_over = False
        self.current = self._new_piece()
        self.next_piece = self._new_piece()
        self.last_drop = time.perf_counter()

    def _new_piece(self) -> Tetromino:
        name = self.rng.choice(list(TETROMINO_SHAPES.keys()))
        start_x = self.width // 2 - 1
        return Tetromino(name=name, rotation=0, x=start_x, y=-1)

    def can_place(self, piece: Tetromino, dx: int = 0, dy: int = 0, rotation: Optional[int] = None) -> bool:
        for px, py in piece.cells(rotation_override=rotation, offset=(dx, dy)):
            if px < 0 or px >= self.width or py >= self.height:
                return False
            if py >= 0 and self.grid[py][px] is not None:
                return False
        return True

    def move(self, dx: int, dy: int) -> bool:
        if self.can_place(self.current, dx, dy):
            self.current.x += dx
            self.current.y += dy
            return True
        return False

    def rotate(self, direction: int = 1) -> bool:
        next_rotation = (self.current.rotation + direction) % len(TETROMINO_SHAPES[self.current.name])
        for shift in (0, -1, 1, -2, 2):
            if self.can_place(self.current, dx=shift, dy=0, rotation=next_rotation):
                self.current.rotation = next_rotation
                self.current.x += shift
                return True
        return False

    def update(self, drop_fast: bool) -> None:
        interval = self.drop_interval * (self.soft_drop_scale if drop_fast else 1.0)
        now = time.perf_counter()
        if now - self.last_drop < interval:
            return
        if not self.move(0, 1):
            self._lock_piece()
        self.last_drop = now

    def _lock_piece(self) -> None:
        for px, py in self.current.cells():
            if 0 <= py < self.height:
                self.grid[py][px] = self.current.name
            else:
                self.game_over = True
        self._clear_lines()
        self.current = self.next_piece
        self.current.x = self.width // 2 - 1
        self.current.y = -1
        self.next_piece = self._new_piece()
        if not self.can_place(self.current):
            self.game_over = True

    def _clear_lines(self) -> None:
        rows = [row for row in self.grid if any(cell is None for cell in row)]
        cleared = self.height - len(rows)
        if cleared <= 0:
            return
        for _ in range(cleared):
            rows.insert(0, [None for _ in range(self.width)])
        self.grid = rows
        self.lines += cleared
        self.score += LINE_SCORES.get(cleared, cleared * 200)
        self.level = 1 + self.lines // 10


def majority_vote(history: Deque[str]) -> str:
    counts = Counter(history)
    return counts.most_common(1)[0][0] if counts else "IDLE"


def connect_lsl(stream_name: str) -> StreamInlet:
    streams = resolve_byprop("name", stream_name, timeout=10)
    if not streams:
        raise RuntimeError(f"Could not find LSL stream named {stream_name}")
    return StreamInlet(streams[0], max_buflen=60)


def init_display() -> Tuple[pygame.Surface, pygame.time.Clock]:
    pygame.init()
    pygame.display.set_caption("MI Tetris")
    screen = pygame.display.set_mode(cfg.SCREEN_SIZE)
    return screen, pygame.time.Clock()


def load_unicode_font(size: int) -> pygame.font.Font:
    candidates = [
        Path("res") / "fonts" / "NotoSansSC-Regular.otf",
        Path("res") / "fonts" / "NotoSansSC-Regular.ttf",
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyh.ttf"),
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


def draw_grid(screen: pygame.Surface, game: TetrisGame) -> None:
    ox, oy = game.board_origin
    cs = game.cell_size
    bg_rect = pygame.Rect(ox - 4, oy - 4, game.width * cs + 8, game.height * cs + 8)
    pygame.draw.rect(screen, (25, 28, 56), bg_rect, border_radius=6)
    for y in range(game.height):
        for x in range(game.width):
            rect = pygame.Rect(ox + x * cs, oy + y * cs, cs - 1, cs - 1)
            color = (35, 40, 70)
            if game.grid[y][x]:
                color = SHAPE_COLORS[game.grid[y][x]]
            pygame.draw.rect(screen, color, rect)
    for x in range(game.width + 1):
        start = (ox + x * cs, oy)
        end = (ox + x * cs, oy + game.height * cs)
        pygame.draw.line(screen, (20, 24, 48), start, end, 1)
    for y in range(game.height + 1):
        start = (ox, oy + y * cs)
        end = (ox + game.width * cs, oy + y * cs)
        pygame.draw.line(screen, (20, 24, 48), start, end, 1)


def draw_active_piece(screen: pygame.Surface, game: TetrisGame) -> None:
    if game.game_over:
        return
    ox, oy = game.board_origin
    cs = game.cell_size
    color = SHAPE_COLORS[game.current.name]
    for x, y in game.current.cells():
        if y < 0:
            continue
        rect = pygame.Rect(ox + x * cs, oy + y * cs, cs - 1, cs - 1)
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, (15, 15, 25), rect, 1)


def draw_next_piece(screen: pygame.Surface, game: TetrisGame, font: pygame.font.Font, anchor: Tuple[int, int]) -> None:
    label = font.render("Next", True, (230, 230, 240))
    screen.blit(label, anchor)
    preview_size = game.cell_size * 4
    rect = pygame.Rect(anchor[0], anchor[1] + label.get_height() + 8, preview_size, preview_size)
    pygame.draw.rect(screen, (24, 28, 54), rect, border_radius=6)
    pattern = TETROMINO_SHAPES[game.next_piece.name][0]
    xs = [dx for dx, _ in pattern]
    ys = [dy for _, dy in pattern]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max_x - min_x + 1
    span_y = max_y - min_y + 1
    cell = max(12, game.cell_size - 6)
    start_x = rect.centerx - (span_x * cell) // 2
    start_y = rect.centery - (span_y * cell) // 2
    for dx, dy in pattern:
        px = start_x + (dx - min_x) * cell
        py = start_y + (dy - min_y) * cell
        block = pygame.Rect(px, py, cell - 2, cell - 2)
        pygame.draw.rect(screen, SHAPE_COLORS[game.next_piece.name], block)


def draw_sidebar(
    screen: pygame.Surface,
    game: TetrisGame,
    fonts: Dict[str, pygame.font.Font],
    command: str,
    confidence: float,
) -> None:
    sidebar_x = game.board_origin[0] + game.width * game.cell_size + 40
    title = fonts["title"].render("MI Tetris", True, (255, 255, 255))
    screen.blit(title, (sidebar_x, 20))
    info_font = fonts["medium"]
    stats = [
        f"Score: {game.score}",
        f"Lines: {game.lines}",
        f"Level: {game.level}",
    ]
    for idx, text in enumerate(stats):
        surf = info_font.render(text, True, (220, 220, 235))
        screen.blit(surf, (sidebar_x, 80 + idx * 26))
    draw_next_piece(screen, game, info_font, (sidebar_x, 170))
    cmd_label = fonts["medium"].render("Command", True, (200, 210, 255))
    screen.blit(cmd_label, (sidebar_x, 340))
    cmd_value = fonts["medium"].render(ACTION_DISPLAY.get(command, command.title()), True, (255, 255, 255))
    screen.blit(cmd_value, (sidebar_x, 370))
    bar_width = 180
    bar_rect = pygame.Rect(sidebar_x, 410, bar_width, 14)
    pygame.draw.rect(screen, (50, 60, 90), bar_rect, border_radius=6)
    filled = int(bar_width * max(0.0, min(1.0, confidence)))
    if filled:
        pygame.draw.rect(screen, (90, 220, 140), pygame.Rect(bar_rect.x, bar_rect.y, filled, bar_rect.height), border_radius=6)
    conf_text = fonts["small"].render(f"Confidence: {confidence:.2f}", True, (200, 220, 255))
    screen.blit(conf_text, (sidebar_x, 430))
    legend_font = fonts["small"]
    legend_lines = [
        "Tongue → rotate",
        "Foot → drop faster",
        "Left hand → move left",
        "Right hand → move right",
        "Space → start/pause",
    ]
    legend_start = 480
    for idx, line in enumerate(legend_lines):
        surf = legend_font.render(line, True, (185, 190, 210))
        screen.blit(surf, (sidebar_x, legend_start + idx * 22))


def draw_scene(screen: pygame.Surface, game: TetrisGame, fonts: Dict[str, pygame.font.Font], command: str, confidence: float) -> None:
    screen.fill((10, 12, 28))
    draw_grid(screen, game)
    draw_active_piece(screen, game)
    draw_sidebar(screen, game, fonts, command, confidence)


def draw_overlay(screen: pygame.Surface, title: str, subtitle: str, title_font: pygame.font.Font, sub_font: pygame.font.Font) -> None:
    overlay = pygame.Surface(cfg.SCREEN_SIZE, pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 170))
    title_surf = title_font.render(title, True, (255, 255, 255))
    sub_surf = sub_font.render(subtitle, True, (220, 220, 230))
    title_rect = title_surf.get_rect(center=(cfg.SCREEN_SIZE[0] // 2, cfg.SCREEN_SIZE[1] // 2 - 30))
    sub_rect = sub_surf.get_rect(center=(cfg.SCREEN_SIZE[0] // 2, title_rect.bottom + 40))
    overlay.blit(title_surf, title_rect)
    overlay.blit(sub_surf, sub_rect)
    screen.blit(overlay, (0, 0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MI-controlled Tetris mini-game")
    parser.add_argument("model_dir", help="Directory containing trained model artifacts")
    parser.add_argument("--stream-name", default=cfg.LSL_STREAM_NAME, help="LSL stream name to resolve")
    parser.add_argument("--drop-interval", type=float, default=0.99, help="Seconds between automatic drops")
    parser.add_argument(
        "--soft-drop-scale",
        type=float,
        default=0.25,
        help="Multiplier applied to drop interval while foot command is active",
    )
    return parser.parse_args()


def run_game(model: FilterBankCSPClassifier, inlet: StreamInlet, args: argparse.Namespace) -> None:
    screen, clock = init_display()
    fonts = {
        "title": pygame.font.SysFont(cfg.FONT_NAME, 34),
        "medium": pygame.font.SysFont(cfg.FONT_NAME, 24),
        "small": pygame.font.SysFont(cfg.FONT_NAME, 18),
    }
    overlay_font = load_unicode_font(56)
    overlay_sub = load_unicode_font(28)
    buffer: List[List[float]] = []
    samples_since_inference = 0
    window_samples = int(cfg.SLIDING_WINDOW_SEC * cfg.SAMPLE_RATE_HZ)
    step_samples = int(cfg.WINDOW_STEP_SEC * cfg.SAMPLE_RATE_HZ)
    max_buffer_samples = window_samples * 3
    history: Deque[str] = deque(maxlen=cfg.MAJORITY_VOTE_WINDOW)
    current_command = "IDLE"
    current_confidence = 0.0
    horizontal_command: Optional[str] = None
    horizontal_elapsed = 0.0
    horizontal_delay = HORIZONTAL_INITIAL_DELAY
    horizontal_single_step = False
    rotate_ready = True
    game = TetrisGame(drop_interval=args.drop_interval, soft_drop_scale=args.soft_drop_scale)
    state = "waiting"
    try:
        while True:
            dt_ms = clock.tick(cfg.REFRESH_RATE_HZ)
            dt_sec = dt_ms / 1000.0 if dt_ms else 0.0
            now = time.perf_counter()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise SystemExit
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    raise SystemExit
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    if state in {"waiting", "finished"}:
                        game.reset()
                        buffer.clear()
                        history.clear()
                        samples_since_inference = 0
                        current_command = "IDLE"
                        current_confidence = 0.0
                        state = "running"
                    elif state == "running":
                        state = "paused"
                    elif state == "paused":
                        state = "running"

            chunk, _ = inlet.pull_chunk(timeout=0.0, max_samples=int(cfg.SAMPLE_RATE_HZ * cfg.CHUNK_LENGTH_SEC))
            if chunk and state == "running":
                trimmed = [sample[: cfg.EXPECTED_CHANNEL_COUNT] for sample in chunk]
                buffer.extend(trimmed)
                samples_since_inference += len(trimmed)
                if len(buffer) > max_buffer_samples:
                    buffer = buffer[-max_buffer_samples:]

            if (
                state == "running"
                and len(buffer) >= window_samples
                and samples_since_inference >= step_samples
            ):
                window = np.asarray(buffer[-window_samples:])
                trial = np.expand_dims(window.T, axis=0)
                probs = model.predict_proba(trial)[0]
                best_idx = int(np.argmax(probs))
                best_label = model.classes_[best_idx]
                best_conf = float(probs[best_idx])
                if best_conf >= cfg.CONFIDENCE_THRESHOLD:
                    history.append(LABEL_TO_ACTION.get(int(best_label), "IDLE"))
                else:
                    history.append("IDLE")
                current_command = majority_vote(history)
                current_confidence = best_conf
                samples_since_inference = 0
                if current_command in {"MOVE_LEFT", "MOVE_RIGHT"}:
                    if current_command != horizontal_command:
                        horizontal_command = current_command
                        horizontal_elapsed = 0.0
                        horizontal_delay = HORIZONTAL_INITIAL_DELAY
                        horizontal_single_step = True
                else:
                    horizontal_command = None
                    horizontal_elapsed = 0.0
                    horizontal_delay = HORIZONTAL_INITIAL_DELAY
                    horizontal_single_step = False

            if state == "running":
                drop_fast = current_command == "SOFT_DROP"
                game.update(drop_fast)
                if horizontal_command and current_command == horizontal_command:
                    step = -1 if horizontal_command == "MOVE_LEFT" else 1
                    if horizontal_single_step:
                        moved = game.move(step, 0)
                        horizontal_single_step = False
                        horizontal_elapsed = 0.0
                        horizontal_delay = HORIZONTAL_INITIAL_DELAY
                        if not moved:
                            horizontal_elapsed = 0.0
                    else:
                        horizontal_elapsed += dt_sec
                        while horizontal_elapsed >= horizontal_delay:
                            moved = game.move(step, 0)
                            horizontal_elapsed -= horizontal_delay
                            horizontal_delay = HORIZONTAL_REPEAT_DELAY
                            if not moved:
                                horizontal_elapsed = 0.0
                                break
                else:
                    horizontal_elapsed = 0.0
                    horizontal_single_step = False
                if current_command == "ROTATE" and rotate_ready:
                    game.rotate()
                    rotate_ready = False
                elif current_command != "ROTATE":
                    rotate_ready = True
                if game.game_over:
                    state = "finished"

            draw_scene(screen, game, fonts, current_command, current_confidence)
            if state == "waiting":
                draw_overlay(screen, "Press Space to start", "Use MI commands to play", overlay_font, overlay_sub)
            elif state == "paused":
                draw_overlay(screen, "Paused", "Press Space to resume", overlay_font, overlay_sub)
            elif state == "finished":
                draw_overlay(screen, "Game over", "Press Space to restart", overlay_font, overlay_sub)
            pygame.display.flip()
    finally:
        pygame.quit()


def main() -> None:
    args = parse_args()
    model = FilterBankCSPClassifier.load(Path(args.model_dir))
    inlet = connect_lsl(args.stream_name)
    run_game(model, inlet, args)


if __name__ == "__main__":
    main()
