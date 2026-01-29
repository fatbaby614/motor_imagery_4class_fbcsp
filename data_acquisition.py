"""Collect motor imagery data from an OpenBCI LSL stream and store MAT files."""
from __future__ import annotations

import argparse
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pygame
from pylsl import StreamInlet, resolve_byprop
from scipy.io import savemat

from config import mi_config as cfg


def cue_text_for_label(label: int) -> str:
    return cfg.EVENT_CUES.get(label, cfg.EVENT_LABELS.get(label, str(label)))


def scale_surface(surface: pygame.Surface, max_size: tuple[int, int]) -> pygame.Surface:
    width, height = surface.get_size()
    max_w, max_h = max_size
    scale = min(max_w / width, max_h / height, 1.0)
    if scale < 1.0:
        new_size = (int(width * scale), int(height * scale))
        return pygame.transform.smoothscale(surface, new_size)
    return surface


def load_cue_images() -> Dict[int, pygame.Surface]:
    images: Dict[int, pygame.Surface] = {}
    base_dir = Path(__file__).resolve().parent
    max_w = int(cfg.SCREEN_SIZE[0] * 0.45)
    max_h = int(cfg.SCREEN_SIZE[1] * 0.45)
    for label, rel_path in cfg.CUE_IMAGE_PATHS.items():
        path = Path(rel_path)
        if not path.is_absolute():
            path = base_dir / path
        if not path.exists():
            print(f"Warning: cue image missing at {path}")
            continue
        image = pygame.image.load(str(path)).convert_alpha()
        images[label] = scale_surface(image, (max_w, max_h))
    return images


def flush_lsl_buffer(inlet: StreamInlet) -> None:
    while True:
        chunk, _ = inlet.pull_chunk(timeout=0.0)
        if not chunk:
            break


def handle_pause_input(paused: bool) -> bool:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            raise SystemExit
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            paused = not paused
    return paused


def draw_pause_overlay(screen: pygame.Surface) -> None:
    overlay = pygame.Surface(cfg.SCREEN_SIZE, pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    font = pygame.font.SysFont(cfg.FONT_NAME, 56)
    text = font.render("Paused", True, (255, 230, 120))
    rect = text.get_rect(center=(cfg.SCREEN_SIZE[0] // 2, cfg.SCREEN_SIZE[1] // 2 - 20))
    overlay.blit(text, rect)
    sub_font = pygame.font.SysFont(cfg.FONT_NAME, 32)
    sub = sub_font.render("Press Space to resume", True, (240, 240, 240))
    sub_rect = sub.get_rect(center=(cfg.SCREEN_SIZE[0] // 2, rect.bottom + 40))
    overlay.blit(sub, sub_rect)
    screen.blit(overlay, (0, 0))


def wait_for_space_to_start(screen: pygame.Surface, clock: pygame.time.Clock) -> None:
    """Render a blocking splash until the user presses space to proceed."""
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                return
        draw_text(
            screen,
            "Ready to Collect Data",
            "Press Space to begin",
            text_color=(255, 255, 255),
            subtext_color=(200, 200, 220),
        )
        pygame.display.flip()
        clock.tick(cfg.REFRESH_RATE_HZ)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect MI data via LSL and store MAT files")
    parser.add_argument("subject_id", type=int, nargs="?", default=2, help="Numeric subject identifier")
    parser.add_argument("session_id", type=int, nargs="?", default=2, help="Numeric session identifier")
    parser.add_argument("trials_per_class", type=int, nargs="?", default=18, help="Number of trials per MI class")
    parser.add_argument("--stream-name", default=cfg.LSL_STREAM_NAME, help="LSL stream name to resolve")
    parser.add_argument(
        "--rest-epochs",
        type=int,
        default=None,
        help="Number of rest epochs to collect (defaults to trials_per_class).",
    )
    return parser.parse_args()


def connect_lsl(stream_name: str) -> StreamInlet:
    print(f"Resolving LSL stream '{stream_name}' ...")
    streams = resolve_byprop("name", stream_name, timeout=10)
    if not streams:
        raise RuntimeError(f"Could not find LSL stream named {stream_name}")
    inlet = StreamInlet(streams[0], max_buflen=60)
    info = inlet.info()
    if info.channel_count() < cfg.EXPECTED_CHANNEL_COUNT:
        raise RuntimeError("Stream has fewer channels than expected")
    print("Stream connected:", info.name(), info.channel_count(), "channels")
    return inlet


def init_pygame() -> pygame.Surface:
    pygame.init()
    pygame.display.set_caption("MI Data Collection")
    screen = pygame.display.set_mode(cfg.SCREEN_SIZE)
    return screen


def draw_text(
    screen: pygame.Surface,
    text: str,
    subtext: str | None = None,
    image: Optional[pygame.Surface] = None,
    text_color: tuple[int, int, int] = (255, 255, 255),
    subtext_color: tuple[int, int, int] = (180, 180, 200),
) -> None:
    screen.fill(cfg.BACKGROUND_COLOR)
    width, height = cfg.SCREEN_SIZE
    center_x = width // 2

    title_y = int(height * 0.18)
    font = pygame.font.SysFont(cfg.FONT_NAME, 52)
    txt = font.render(text, True, text_color)
    rect = txt.get_rect(center=(center_x, title_y))
    screen.blit(txt, rect)

    if subtext:
        sub_font = pygame.font.SysFont(cfg.FONT_NAME, 30)
        sub_txt = sub_font.render(subtext, True, subtext_color)
        sub_rect = sub_txt.get_rect(center=(center_x, rect.bottom + 35))
        screen.blit(sub_txt, sub_rect)

    if image is not None:
        max_w = int(width * 0.5)
        max_h = int(height * 0.5)
        if image.get_width() > max_w or image.get_height() > max_h:
            image = scale_surface(image, (max_w, max_h))
        img_rect = image.get_rect(center=(center_x, int(height * 0.65)))
        screen.blit(image, img_rect)


def wait_with_ui(
    duration_sec: float,
    screen: pygame.Surface,
    clock: pygame.time.Clock,
    text: str,
    subtext: str,
    image: Optional[pygame.Surface] = None,
    show_countdown: bool = False,
    text_color: tuple[int, int, int] = (255, 255, 255),
    subtext_color: tuple[int, int, int] = (180, 180, 200),
) -> None:
    elapsed = 0.0
    paused = False
    while elapsed < duration_sec:
        paused = handle_pause_input(paused)
        if paused:
            draw_pause_overlay(screen)
            pygame.display.flip()
            clock.tick(cfg.REFRESH_RATE_HZ)
            continue
        countdown_text = subtext
        if show_countdown:
            remaining = math.ceil(max(0.0, duration_sec - elapsed))
            countdown_text = f"{subtext} ({remaining:d}s)"
        draw_text(screen, text, countdown_text, image=image, text_color=text_color, subtext_color=subtext_color)
        pygame.display.flip()
        dt_ms = clock.tick(cfg.REFRESH_RATE_HZ)
        elapsed += dt_ms / 1000.0


def record_epoch(
    inlet: StreamInlet,
    samples_needed: int,
    screen: pygame.Surface,
    clock: pygame.time.Clock,
    cue_text: str,
    instruction: str = "Imagine performing the movement",
    image: Optional[pygame.Surface] = None,
    show_countdown: bool = False,
    text_color: tuple[int, int, int] = (255, 255, 255),
    subtext_color: tuple[int, int, int] = (180, 180, 200),
) -> np.ndarray:
    collected: List[List[float]] = []
    paused = False
    target_duration = samples_needed / cfg.SAMPLE_RATE_HZ
    elapsed_time = 0.0
    while len(collected) < samples_needed:
        paused = handle_pause_input(paused)
        if paused:
            draw_pause_overlay(screen)
            pygame.display.flip()
            clock.tick(cfg.REFRESH_RATE_HZ)
            continue
        subtitle = instruction
        if show_countdown:
            remaining_sec = math.ceil(max(0.0, target_duration - elapsed_time))
            subtitle = f"{instruction} ({remaining_sec:d}s)"
        draw_text(screen, cue_text, subtitle, image=image, text_color=text_color, subtext_color=subtext_color)
        pygame.display.flip()
        chunk, _ = inlet.pull_chunk(timeout=0.0, max_samples=int(cfg.SAMPLE_RATE_HZ * cfg.CHUNK_LENGTH_SEC))
        if chunk:
            collected.extend(chunk)
        dt_ms = clock.tick(cfg.REFRESH_RATE_HZ)
        elapsed_time += dt_ms / 1000.0
    data = np.asarray(collected[:samples_needed])
    return data[:, : cfg.EXPECTED_CHANNEL_COUNT]


def build_trial_schedule(trials_per_class: int) -> List[int]:
    labels = [1, 2, 3, 4]  # UP, DOWN, LEFT, RIGHT
    schedule = []
    for lbl in labels:
        schedule.extend([lbl] * trials_per_class)
    random.shuffle(schedule)
    return schedule


def main() -> None:
    args = parse_args()
    inlet = connect_lsl(args.stream_name)
    screen = init_pygame()
    clock = pygame.time.Clock()
    cue_images = load_cue_images()

    wait_for_space_to_start(screen, clock)

    samples_per_epoch = int(cfg.EPOCH_DURATION_SEC * cfg.SAMPLE_RATE_HZ)
    rest_samples = int(cfg.REST_EPOCH_DURATION_SEC * cfg.SAMPLE_RATE_HZ)
    baseline_samples = int(cfg.BASELINE_DURATION_SEC * cfg.SAMPLE_RATE_HZ)
    trial_schedule = build_trial_schedule(args.trials_per_class)

    dataset = []
    labels = []
    timestamps: List[Dict[str, float]] = []

    wait_with_ui(
        cfg.PRE_EVENT_MARGIN_SEC,
        screen,
        clock,
        "Baseline Incoming",
        "Stay relaxed for calibration",
        image=cue_images.get(0),
        show_countdown=True,
        text_color=(80, 200, 120),
        subtext_color=(80, 200, 120),
    )
    baseline_start = datetime.now(timezone.utc).timestamp()
    flush_lsl_buffer(inlet)
    baseline_epoch = record_epoch(
        inlet,
        baseline_samples,
        screen,
        clock,
        cue_text_for_label(0),
        "Stay still and relax",
        image=cue_images.get(0),
        show_countdown=True,
        text_color=(80, 200, 120),
        subtext_color=(80, 200, 120),
    )
    baseline_data = baseline_epoch.T  # (channels, samples)
    baseline_segment_len = samples_per_epoch
    num_segments = baseline_data.shape[1] // baseline_segment_len
    if num_segments == 0:
        padded = np.zeros((baseline_data.shape[0], baseline_segment_len))
        padded[:, : baseline_data.shape[1]] = baseline_data
        baseline_segments = [(baseline_start, padded)]
    else:
        baseline_segments = []
        for seg_idx in range(num_segments):
            start_idx = seg_idx * baseline_segment_len
            end_idx = start_idx + baseline_segment_len
            onset = baseline_start + (start_idx / cfg.SAMPLE_RATE_HZ)
            baseline_segments.append((onset, baseline_data[:, start_idx:end_idx]))
    for onset, segment in baseline_segments:
        dataset.append(segment)
        labels.append(0)
        timestamps.append({"onset_utc": onset, "label": 0, "segment": "baseline"})

    baseline_rest_epochs = len(baseline_segments)
    rest_target = args.rest_epochs if args.rest_epochs is not None else args.trials_per_class
    if rest_target < 0:
        print("Warning: rest epoch target was negative, ignoring override.")
        rest_target = 0
    rest_blocks = max(0, rest_target - baseline_rest_epochs)
    if rest_blocks:
        print(
            f"Baseline contributed {baseline_rest_epochs} rest epochs; "
            f"collecting {rest_blocks} more to reach target {rest_target}."
        )
    else:
        print(
            f"Baseline already covers desired rest count (target={rest_target}, baseline={baseline_rest_epochs})."
        )
    if rest_blocks:
        for rest_idx in range(1, rest_blocks + 1):
            wait_with_ui(
                1.0,
                screen,
                clock,
                "Collecting Rest",
                f"Rest sample {rest_idx}/{rest_blocks}",
                image=cue_images.get(0),
                show_countdown=False,
                text_color=(80, 200, 120),
                subtext_color=(80, 200, 120),
            )
            rest_start = datetime.now(timezone.utc).timestamp()
            flush_lsl_buffer(inlet)
            rest_epoch = record_epoch(
                inlet,
                rest_samples,
                screen,
                clock,
                cue_text_for_label(0),
                "Relax and stay still",
                image=cue_images.get(0),
                show_countdown=True,
                text_color=(80, 200, 120),
                subtext_color=(80, 200, 120),
            )
            dataset.append(rest_epoch.T)
            labels.append(0)
            timestamps.append({"onset_utc": rest_start, "label": 0, "segment": "bulk_rest"})

    for idx, label in enumerate(trial_schedule, start=1):
        cue_text = cue_text_for_label(label)
        cue_image = cue_images.get(label)
        wait_with_ui(
            cfg.INTER_TRIAL_INTERVAL_SEC,
            screen,
            clock,
            f"Trial {idx}/{len(trial_schedule)}",
            f"Next: {cue_text}",
            image=cue_image,
            show_countdown=True,
            text_color=(80, 200, 120),
            subtext_color=(80, 200, 120),
        )
        wait_with_ui(
            cfg.PRE_EVENT_MARGIN_SEC,
            screen,
            clock,
            "Get Ready",
            f"Prepare for {cue_text}",
            image=cue_image,
            show_countdown=True,
            text_color=(255, 165, 0),
            subtext_color=(255, 200, 120),
        )
        start_time = datetime.now(timezone.utc).timestamp()
        flush_lsl_buffer(inlet)
        epoch = record_epoch(
            inlet,
            samples_per_epoch,
            screen,
            clock,
            cue_text,
            "Imagine the indicated movement",
            image=cue_image,
            show_countdown=True,
            text_color=(220, 80, 80),
            subtext_color=(255, 180, 180),
        )
        dataset.append(epoch.T)
        labels.append(label)
        timestamps.append({"onset_utc": start_time, "label": int(label)})

    pygame.quit()

    data_array = np.stack(dataset)
    labels_array = np.asarray(labels)
    metadata = {
        "channels": cfg.CHANNEL_LABELS,
        "sample_rate": cfg.SAMPLE_RATE_HZ,
        "labels": cfg.EVENT_LABELS,
        "timestamps": timestamps,
    }

    cfg.DATA_ROOT.mkdir(parents=True, exist_ok=True)
    mat_path = cfg.DATA_ROOT / cfg.MAT_FILE_TEMPLATE.format(
        subject_id=args.subject_id,
        session_id=args.session_id,
    )
    savemat(
        mat_path,
        {
            "data": data_array,
            "labels": labels_array,
            "metadata": metadata,
        },
    )
    print(f"Saved dataset to {mat_path}")


if __name__ == "__main__":
    main()
