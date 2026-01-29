"""Real-time motor imagery control with LSL streaming and pygame UI."""
from __future__ import annotations

import argparse
from collections import Counter, deque
from pathlib import Path
from typing import Deque, List, Tuple

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

COMMAND_TO_VECTOR = {
    cfg.IDLE_COMMAND: (0.0, 0.0),
    "UP": (0.0, -1.0),
    "DOWN": (0.0, 1.0),
    "LEFT": (-1.0, 0.0),
    "RIGHT": (1.0, 0.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time MI control loop")
    parser.add_argument("model_dir", help="Directory containing trained model artifacts")
    parser.add_argument("--stream-name", default=cfg.LSL_STREAM_NAME, help="LSL stream name")
    return parser.parse_args()


def connect_lsl(stream_name: str) -> StreamInlet:
    streams = resolve_byprop("name", stream_name, timeout=10)
    if not streams:
        raise RuntimeError(f"Could not find LSL stream named {stream_name}")
    inlet = StreamInlet(streams[0], max_buflen=60)
    return inlet


def init_pygame() -> Tuple[pygame.Surface, pygame.time.Clock]:
    pygame.init()
    pygame.display.set_caption("MI Real-Time Control")
    screen = pygame.display.set_mode(cfg.SCREEN_SIZE)
    clock = pygame.time.Clock()
    return screen, clock


def draw_ui(screen: pygame.Surface, command: str, confidence: float, cursor_pos: np.ndarray) -> None:
    screen.fill(cfg.BACKGROUND_COLOR)
    font = pygame.font.SysFont(cfg.FONT_NAME, 64)
    text = font.render(f"Command: {command}", True, (255, 255, 255))
    rect = text.get_rect(center=(cfg.SCREEN_SIZE[0] // 2, cfg.SCREEN_SIZE[1] // 2 - 60))
    screen.blit(text, rect)

    sub_font = pygame.font.SysFont(cfg.FONT_NAME, 30)
    bar_width = int(cfg.SCREEN_SIZE[0] * 0.65)
    bar_height = 12
    bar_x = (cfg.SCREEN_SIZE[0] - bar_width) // 2
    bar_y = cfg.SCREEN_SIZE[1] - 50
    label_text = sub_font.render("Confidence", True, (180, 220, 255))
    label_rect = label_text.get_rect(midbottom=(cfg.SCREEN_SIZE[0] // 2, bar_y - 6))
    screen.blit(label_text, label_rect)
    pygame.draw.rect(screen, (60, 60, 90), (bar_x, bar_y, bar_width, bar_height), border_radius=6)
    filled_width = int(bar_width * max(0.0, min(1.0, confidence)))
    if filled_width > 0:
        pygame.draw.rect(screen, (80, 220, 120), (bar_x, bar_y, filled_width, bar_height), border_radius=6)
    value_text = sub_font.render(f"{confidence:.2f}", True, (255, 255, 255))
    value_rect = value_text.get_rect(midtop=(cfg.SCREEN_SIZE[0] // 2, bar_y + bar_height + 6))
    screen.blit(value_text, value_rect)

    pygame.draw.circle(screen, (80, 100, 140), (cfg.SCREEN_SIZE[0] // 2, cfg.SCREEN_SIZE[1] // 2 + 120), 60, width=1)
    pygame.draw.circle(screen, cfg.CURSOR_COLOR, cursor_pos.astype(int), cfg.CURSOR_RADIUS)


def majority_vote(history: Deque[str]) -> str:
    counts = Counter(history)
    if not counts:
        return cfg.IDLE_COMMAND
    return counts.most_common(1)[0][0]


def run_loop(model: FilterBankCSPClassifier, inlet: StreamInlet) -> None:
    screen, clock = init_pygame()
    buffer: List[List[float]] = []
    samples_since_inference = 0
    window_samples = int(cfg.SLIDING_WINDOW_SEC * cfg.SAMPLE_RATE_HZ)
    step_samples = int(cfg.WINDOW_STEP_SEC * cfg.SAMPLE_RATE_HZ)
    max_buffer_samples = window_samples * 3
    prediction_history: Deque[str] = deque(maxlen=cfg.MAJORITY_VOTE_WINDOW)
    current_command = cfg.IDLE_COMMAND
    current_confidence = 0.0
    cursor_pos = np.array([cfg.SCREEN_SIZE[0] / 2, cfg.SCREEN_SIZE[1] / 2], dtype=float)
    cursor_vel = np.zeros(2, dtype=float)

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise SystemExit
            chunk, _ = inlet.pull_chunk(timeout=0.0, max_samples=int(cfg.SAMPLE_RATE_HZ * cfg.CHUNK_LENGTH_SEC))
            if chunk:
                buffer.extend(sample[: cfg.EXPECTED_CHANNEL_COUNT] for sample in chunk)
                samples_since_inference += len(chunk)
                if len(buffer) > max_buffer_samples:
                    buffer = buffer[-max_buffer_samples:]

            if len(buffer) >= window_samples and samples_since_inference >= step_samples:
                window = np.asarray(buffer[-window_samples:])
                trial = np.expand_dims(window.T, axis=0)
                probs = model.predict_proba(trial)[0]
                best_idx = int(np.argmax(probs))
                best_label = model.classes_[best_idx]
                best_conf = float(probs[best_idx])
                if best_conf >= cfg.CONFIDENCE_THRESHOLD:
                    prediction_history.append(LABEL_TO_COMMAND.get(int(best_label), cfg.IDLE_COMMAND))
                else:
                    prediction_history.append(cfg.IDLE_COMMAND)
                current_command = majority_vote(prediction_history)
                current_confidence = best_conf
                samples_since_inference = 0

            direction = np.array(COMMAND_TO_VECTOR.get(current_command, (0.0, 0.0)), dtype=float)
            target_velocity = direction * cfg.CURSOR_SPEED_PX
            cursor_vel = (
                cfg.CURSOR_DAMPING * cursor_vel
                + (1.0 - cfg.CURSOR_DAMPING) * target_velocity
            )
            cursor_pos += cursor_vel
            cursor_pos[0] = np.clip(cursor_pos[0], cfg.CURSOR_RADIUS, cfg.SCREEN_SIZE[0] - cfg.CURSOR_RADIUS)
            cursor_pos[1] = np.clip(cursor_pos[1], cfg.CURSOR_RADIUS, cfg.SCREEN_SIZE[1] - cfg.CURSOR_RADIUS)

            draw_ui(screen, current_command, current_confidence, cursor_pos)
            pygame.display.flip()
            clock.tick(cfg.REFRESH_RATE_HZ)
    finally:
        pygame.quit()


def main() -> None:
    args = parse_args()
    model = FilterBankCSPClassifier.load(Path(args.model_dir))
    inlet = connect_lsl(args.stream_name)
    run_loop(model, inlet)


if __name__ == "__main__":
    main()
