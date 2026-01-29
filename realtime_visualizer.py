"""Real-time visualization of EEG channels (time + frequency domain) using pygame."""
from __future__ import annotations

import argparse
import math
from collections import deque
from typing import Deque, Dict, List, Sequence, Tuple

import numpy as np
import pygame
from pylsl import StreamInlet, resolve_byprop

from config import mi_config as cfg

CHANNEL_INDICES = [0, 1, 2, 7]  # 1-based channels 1,2,3,8 -> zero-based indices
CHANNEL_LABELS = [cfg.CHANNEL_LABELS[i] if i < len(cfg.CHANNEL_LABELS) else f"Ch{i+1}" for i in CHANNEL_INDICES]
CHANNEL_COLORS: Sequence[Tuple[int, int, int]] = [
    (255, 120, 120),
    (120, 200, 255),
    (255, 200, 120),
    (160, 255, 140),
]
TIME_WINDOW_SEC = 1.5
FFT_MAX_FREQ = 60.0
FREQ_MIN = 4.0
BAND_COLORS = {
    "Delta": (100, 100, 150),
    "Theta": (80, 130, 180),
    "Alpha": (100, 170, 140),
    "Beta": (180, 120, 80),
}
BANDS = [
    (1.0, 4.0, "Delta"),
    (4.0, 8.0, "Theta"),
    (8.0, 13.0, "Alpha"),
    (13.0, 30.0, "Beta"),
]
MU_BETA_BANDS = [
    (8.0, 13.0, "Mu"),
    (13.0, 30.0, "Beta"),
]
PADDING = 40
NOTCH_FREQ = 50.0
NOTCH_WIDTH = 2.0
PSD_ROW_HEIGHT = 110
GENERAL_BAR_HEIGHT = 9
GENERAL_BAR_GAP = 3
PSD_BAR_HEIGHT = 11
PSD_BAR_GAP = 3
BAND_LABEL_WIDTH = 52
PSD_LABEL_WIDTH = 66
BAR_VALUE_COLOR = (235, 235, 245)
PSD_BAR_COLORS = {
    "Mu": (160, 210, 255),
    "Beta": (255, 190, 140),
}
PSD_GRID_LAYOUT = {
    "C3": (0, 0),
    "C4": (1, 0),
    "Cz": (0, 1),
    "FC1": (1, 1),
    "FC2": (0, 2),
    "T3": (1, 2),
    "T4": (0, 3),
    "Fz": (1, 3),
}
PSD_GRID_COLS = 2
PSD_GRID_ROWS = max((pos[1] for pos in PSD_GRID_LAYOUT.values()), default=-1) + 1
SLIDER_AREA_HEIGHT = 60
SLIDER_BAR_WIDTH = 260
SLIDER_BAR_HEIGHT = 8
SLIDER_KNOB_WIDTH = 14
SLIDER_COLOR = (120, 170, 255)
SLIDER_BG = (60, 70, 100)
SLIDER_LABEL_COLOR = (220, 220, 235)
X_ZOOM_RANGE = (0.4, 2.5)
Y_ZOOM_RANGE = (0.5, 3.0)
Y_SLIDER_WIDTH = 80
Y_SLIDER_GAP = 12
Y_SLIDER_PAD = 12
Y_SLIDER_BAR_WIDTH = 12
Y_SLIDER_KNOB_HEIGHT = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time EEG visualizer (time + frequency domain)")
    parser.add_argument("--stream-name", default=cfg.LSL_STREAM_NAME, help="LSL EEG stream name")
    parser.add_argument("--buffer-seconds", type=float, default=TIME_WINDOW_SEC, help="History window (seconds)")
    parser.add_argument("--max-freq", type=float, default=FFT_MAX_FREQ, help="Max frequency to display (Hz)")
    return parser.parse_args()


def connect_lsl(stream_name: str) -> StreamInlet:
    streams = resolve_byprop("name", stream_name, timeout=10)
    if not streams:
        raise RuntimeError(f"Could not find LSL stream named {stream_name}")
    return StreamInlet(streams[0], max_buflen=60)


def init_pygame() -> Tuple[pygame.Surface, pygame.time.Clock]:
    pygame.init()
    pygame.display.set_caption("EEG Time/Freq Visualizer")
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    return screen, pygame.time.Clock()


def draw_axes(surface: pygame.Surface, rect: pygame.Rect) -> None:
    pygame.draw.rect(surface, (60, 60, 80), rect, width=1)


def slider_value_to_pos(value: float, vmin: float, vmax: float, bar_rect: pygame.Rect) -> int:
    ratio = (value - vmin) / max(1e-6, (vmax - vmin))
    ratio = max(0.0, min(1.0, ratio))
    return int(bar_rect.left + ratio * bar_rect.width)


def slider_pos_to_value(x: int, vmin: float, vmax: float, bar_rect: pygame.Rect) -> float:
    ratio = (x - bar_rect.left) / max(1e-6, bar_rect.width)
    ratio = max(0.0, min(1.0, ratio))
    return vmin + ratio * (vmax - vmin)


def slider_value_to_pos_vertical(value: float, vmin: float, vmax: float, bar_rect: pygame.Rect) -> int:
    ratio = (value - vmin) / max(1e-6, (vmax - vmin))
    ratio = max(0.0, min(1.0, ratio))
    return int(bar_rect.bottom - ratio * bar_rect.height)


def slider_pos_to_value_vertical(y: int, vmin: float, vmax: float, bar_rect: pygame.Rect) -> float:
    ratio = (bar_rect.bottom - y) / max(1e-6, bar_rect.height)
    ratio = max(0.0, min(1.0, ratio))
    return vmin + ratio * (vmax - vmin)


def draw_slider(
    screen: pygame.Surface,
    rect: pygame.Rect,
    label: str,
    value: float,
    vmin: float,
    vmax: float,
    active: bool,
) -> Tuple[pygame.Rect, pygame.Rect]:
    pygame.draw.rect(screen, SLIDER_BG, rect, border_radius=6)
    bar_height = SLIDER_BAR_HEIGHT
    bar_width = SLIDER_BAR_WIDTH
    bar_left = rect.centerx - bar_width // 2
    bar_top = rect.centery - bar_height // 2
    bar_rect = pygame.Rect(bar_left, bar_top, bar_width, bar_height)
    pygame.draw.rect(screen, (40, 50, 80), bar_rect, border_radius=3)
    knob_x = slider_value_to_pos(value, vmin, vmax, bar_rect)
    knob_rect = pygame.Rect(knob_x - SLIDER_KNOB_WIDTH // 2, bar_rect.top - 6, SLIDER_KNOB_WIDTH, bar_height + 12)
    pygame.draw.rect(screen, SLIDER_COLOR if active else (150, 180, 210), knob_rect, border_radius=4)
    pygame.draw.rect(screen, (30, 30, 50), bar_rect, 1, border_radius=3)
    pygame.draw.rect(screen, (20, 20, 40), knob_rect, 1, border_radius=4)
    font = pygame.font.SysFont(cfg.FONT_NAME, 14)
    label_surface = font.render(f"{label}: {value:.2f}x", True, SLIDER_LABEL_COLOR)
    screen.blit(label_surface, (bar_left, rect.top + 4))
    return knob_rect, bar_rect


def draw_vertical_slider(
    screen: pygame.Surface,
    rect: pygame.Rect,
    label: str,
    value: float,
    vmin: float,
    vmax: float,
    active: bool,
) -> Tuple[pygame.Rect, pygame.Rect]:
    pygame.draw.rect(screen, SLIDER_BG, rect, border_radius=8)
    font = pygame.font.SysFont(cfg.FONT_NAME, 14)
    label_surface = font.render(label, True, SLIDER_LABEL_COLOR)
    value_surface = font.render(f"{value:.2f}x", True, SLIDER_LABEL_COLOR)
    label_pos = (rect.centerx - label_surface.get_width() // 2, rect.top + 6)
    screen.blit(label_surface, label_pos)
    value_pos = (rect.centerx - value_surface.get_width() // 2, label_pos[1] + label_surface.get_height() + 2)
    screen.blit(value_surface, value_pos)
    bar_top = value_pos[1] + value_surface.get_height() + Y_SLIDER_PAD
    bar_height = rect.bottom - Y_SLIDER_PAD - bar_top
    bar_height = max(40, bar_height)
    bar_rect = pygame.Rect(
        rect.centerx - Y_SLIDER_BAR_WIDTH // 2,
        bar_top,
        Y_SLIDER_BAR_WIDTH,
        bar_height,
    )
    pygame.draw.rect(screen, (40, 50, 80), bar_rect, border_radius=4)
    knob_y = slider_value_to_pos_vertical(value, vmin, vmax, bar_rect)
    knob_rect = pygame.Rect(
        bar_rect.left - 6,
        knob_y - Y_SLIDER_KNOB_HEIGHT // 2,
        bar_rect.width + 12,
        Y_SLIDER_KNOB_HEIGHT,
    )
    pygame.draw.rect(screen, SLIDER_COLOR if active else (150, 180, 210), knob_rect, border_radius=4)
    pygame.draw.rect(screen, (20, 20, 40), bar_rect, 1, border_radius=4)
    pygame.draw.rect(screen, (30, 30, 50), knob_rect, 1, border_radius=4)
    return knob_rect, bar_rect


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    if signal.size == 0:
        return signal
    max_val = np.max(np.abs(signal))
    if max_val < 1e-6:
        return np.zeros_like(signal)
    return signal / max_val


def standardize_signal(signal: np.ndarray) -> np.ndarray:
    if signal.size == 0:
        return signal
    centered = signal - np.mean(signal)
    std = np.std(centered)
    if std < 1e-6:
        return np.zeros_like(centered)
    return centered / std


def compute_fft(data: np.ndarray, sample_rate: float, max_freq: float) -> Tuple[np.ndarray, np.ndarray]:
    if data.size == 0:
        return np.array([]), np.array([])
    window = np.hanning(data.shape[1])
    spectrum = np.fft.rfft(data * window, axis=1)
    freqs = np.fft.rfftfreq(data.shape[1], d=1.0 / sample_rate)
    notch_mask = np.abs(freqs - NOTCH_FREQ) <= (NOTCH_WIDTH / 2.0)
    if np.any(notch_mask):
        spectrum[:, notch_mask] = 0.0
    low_cut = max(FREQ_MIN, 0.0)
    mask = (freqs >= low_cut) & (freqs <= max_freq)
    return freqs[mask], np.abs(spectrum[:, mask])


def band_powers(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    bands: Sequence[Tuple[float, float, str]] = BANDS,
) -> Dict[str, List[float]]:
    powers: Dict[str, List[float]] = {name: [] for *_ , name in bands}
    if freqs.size == 0:
        for name in powers:
            powers[name] = [0.0] * len(CHANNEL_INDICES)
        return powers
    for low, high, name in bands:
        mask = (freqs >= low) & (freqs < high)
        if np.any(mask):
            band_window = spectrum[:, mask]
            band_power = np.square(band_window).mean(axis=1)
        else:
            band_power = np.zeros(len(CHANNEL_INDICES))
        powers[name] = band_power.tolist()
    return powers


def draw_time_domain(
    screen: pygame.Surface,
    rect: pygame.Rect,
    data: np.ndarray,
    x_zoom: float,
    y_zoom: float,
) -> None:
    draw_axes(screen, rect)
    if data.size == 0:
        return
    samples = data.shape[1]
    x_scale = (rect.width / max(1, samples - 1)) * max(0.1, x_zoom)
    channel_height = rect.height / len(CHANNEL_INDICES)
    per_row_amp = channel_height / 2 - 6
    global_amp = max(8.0, min(per_row_amp * 1.5, channel_height * 0.9)) * max(0.1, y_zoom)
    for ch_idx, signal in enumerate(data):
        color = CHANNEL_COLORS[ch_idx % len(CHANNEL_COLORS)]
        norm = standardize_signal(signal)
        row_top = rect.top + ch_idx * channel_height
        row_mid = row_top + channel_height / 2
        amp = global_amp
        points = [
            (rect.left + i * x_scale, row_mid - norm[i] * amp)
            for i in range(samples)
        ]
        if len(points) > 1:
            pygame.draw.lines(screen, color, False, points, 2)
        label = pygame.font.SysFont(cfg.FONT_NAME, 20).render(CHANNEL_LABELS[ch_idx], True, color)
        screen.blit(label, (rect.left + 8, row_top + 6))


def draw_frequency_domain(screen: pygame.Surface, rect: pygame.Rect, freqs: np.ndarray, spectrum: np.ndarray) -> None:
    draw_axes(screen, rect)
    if freqs.size == 0:
        return
    max_amp = np.max(spectrum) if spectrum.size else 1.0
    max_amp = max(max_amp, 1e-6)
    freq_span = max(freqs[-1] - freqs[0], 1e-6)
    for ch_idx in range(spectrum.shape[0]):
        color = CHANNEL_COLORS[ch_idx % len(CHANNEL_COLORS)]
        normalized = spectrum[ch_idx] / max_amp
        points = []
        for idx, value in enumerate(normalized):
            if len(normalized) == 1:
                x_ratio = 0.0
                x = rect.left
            else:
                x_ratio = (freqs[idx] - freqs[0]) / freq_span
                x = rect.left + max(0.0, min(1.0, x_ratio)) * rect.width
            y = rect.bottom - value * (rect.height - 20)
            points.append((x, y))
        if len(points) > 1:
            pygame.draw.lines(screen, color, False, points, 2)
    font = pygame.font.SysFont(cfg.FONT_NAME, 18)
    label = font.render(f"{freqs[0]:.1f}-{freqs[-1]:.1f} Hz", True, (200, 200, 220))
    screen.blit(label, (rect.right - label.get_width() - 8, rect.top + 6))
    tick_font = pygame.font.SysFont(cfg.FONT_NAME, 14)
    axis_y = rect.bottom
    tick_start = int(math.ceil(freqs[0] / 10.0) * 10)
    tick_end = int(math.floor(freqs[-1] / 10.0) * 10)
    if tick_start <= tick_end:
        for tick in range(tick_start, tick_end + 1, 10):
            ratio = (tick - freqs[0]) / freq_span
            ratio = max(0.0, min(1.0, ratio))
            x = int(rect.left + ratio * rect.width)
            pygame.draw.line(screen, (90, 90, 120), (x, axis_y), (x, axis_y + 6), 1)
            tick_label = tick_font.render(f"{tick}", True, (200, 200, 220))
            screen.blit(tick_label, (x - tick_label.get_width() // 2, axis_y + 8))
    marker_font = pygame.font.SysFont(cfg.FONT_NAME, 18)
    for low, high, name in BANDS:
        clipped_low = max(low, freqs[0])
        clipped_high = min(high, freqs[-1])
        if clipped_high <= clipped_low:
            continue
        start_ratio = (clipped_low - freqs[0]) / freq_span
        end_ratio = (clipped_high - freqs[0]) / freq_span
        rect_left = rect.left + start_ratio * rect.width
        rect_width = (end_ratio - start_ratio) * rect.width
        overlay = pygame.Surface((rect_width, rect.height), pygame.SRCALPHA)
        overlay.fill((*BAND_COLORS.get(name, (80, 80, 80)), 40))
        screen.blit(overlay, (rect_left, rect.top))
        pygame.draw.rect(
            screen,
            BAND_COLORS.get(name, (120, 120, 120)),
            (rect_left, rect.top, rect_width, rect.height),
            1,
        )
        marker_txt = marker_font.render(name, True, (220, 220, 240))
        screen.blit(marker_txt, (rect_left + 4, rect.top + 4))


def draw_band_panels(
    screen: pygame.Surface,
    rect: pygame.Rect,
    band_power: Dict[str, List[float]],
    mu_beta_power: Dict[str, List[float]],
) -> None:
    title_font = pygame.font.SysFont(cfg.FONT_NAME, 16)
    label_font = pygame.font.SysFont(cfg.FONT_NAME, 14)
    value_font = pygame.font.SysFont(cfg.FONT_NAME, 13)
    row_height = PSD_ROW_HEIGHT
    general_names = [name for *_ , name in BANDS]
    general_values: Dict[str, List[float]] = {}
    for _, _, name in BANDS:
        values = list(band_power.get(name, [0.0] * len(CHANNEL_INDICES)))
        if len(values) < len(CHANNEL_INDICES):
            values.extend([0.0] * (len(CHANNEL_INDICES) - len(values)))
        general_values[name] = values
    general_max = max(
        (max(vals) if vals else 0.0) for vals in general_values.values()
    ) if general_values else 0.0
    general_max = max(general_max, 1e-6)

    mu_values = list(mu_beta_power.get("Mu", [0.0] * len(CHANNEL_INDICES)))
    beta_values = list(mu_beta_power.get("Beta", [0.0] * len(CHANNEL_INDICES)))
    if len(mu_values) < len(CHANNEL_INDICES):
        mu_values.extend([0.0] * (len(CHANNEL_INDICES) - len(mu_values)))
    if len(beta_values) < len(CHANNEL_INDICES):
        beta_values.extend([0.0] * (len(CHANNEL_INDICES) - len(beta_values)))
    psd_values = mu_values + beta_values
    psd_max = max(psd_values) if psd_values else 0.0
    psd_max = max(psd_max, 1e-6)

    legend_font = pygame.font.SysFont(cfg.FONT_NAME, 14)
    legend_text = legend_font.render("Band power & μ/β PSD (horizontal bars)", True, (210, 210, 230))
    screen.blit(legend_text, (rect.left + 10, rect.top + 4))
    content_top = rect.top + legend_text.get_height() + 8
    inner_margin = 16
    col_width = (rect.width - inner_margin * 2) / PSD_GRID_COLS

    psd_rows = [
        ("Mu PSD", "Mu", mu_values, PSD_BAR_COLORS.get("Mu", (160, 210, 255))),
        ("Beta PSD", "Beta", beta_values, PSD_BAR_COLORS.get("Beta", (255, 190, 140))),
    ]

    for ch_idx, label in enumerate(CHANNEL_LABELS):
        col, row = PSD_GRID_LAYOUT.get(label, (ch_idx % PSD_GRID_COLS, ch_idx // PSD_GRID_COLS))
        cell_left = int(rect.left + inner_margin + col * col_width)
        cell_top = int(content_top + row * row_height)
        title_surface = title_font.render(label, True, CHANNEL_COLORS[ch_idx % len(CHANNEL_COLORS)])
        screen.blit(title_surface, (cell_left, cell_top))

        general_top = cell_top + title_surface.get_height() + 2
        general_bar_start = cell_left + BAND_LABEL_WIDTH
        general_bar_max_width = max(30, int(col_width - (general_bar_start - cell_left) - 30))
        for idx, name in enumerate(general_names):
            value = general_values.get(name, [0.0] * len(CHANNEL_INDICES))[ch_idx]
            bar_y = general_top + idx * (GENERAL_BAR_HEIGHT + GENERAL_BAR_GAP)
            label_surface = label_font.render(name, True, BAND_COLORS.get(name, (180, 180, 200)))
            screen.blit(label_surface, (cell_left, bar_y - 2))
            bar_x = general_bar_start
            width = int(round((value / general_max) * general_bar_max_width)) if general_max > 0 else 0
            width = max(1, width)
            bar_rect = pygame.Rect(bar_x, int(bar_y), width, GENERAL_BAR_HEIGHT)
            color = BAND_COLORS.get(name, (120, 160, 200))
            pygame.draw.rect(screen, color, bar_rect)
            pygame.draw.rect(screen, (30, 30, 50), bar_rect, 1)
            value_surface = value_font.render(f"{value:.0f}", True, BAR_VALUE_COLOR)
            value_x = min(cell_left + int(col_width) - value_surface.get_width() - 4, bar_rect.right + 4)
            value_y = max(cell_top, bar_rect.top - 2)
            screen.blit(value_surface, (value_x, value_y))

        psd_top = general_top + len(general_names) * (GENERAL_BAR_HEIGHT + GENERAL_BAR_GAP) + 6
        psd_bar_start = cell_left + PSD_LABEL_WIDTH
        psd_bar_max_width = max(30, int(col_width - (psd_bar_start - cell_left) - 30))
        for idx, (disp_name, key, values, color) in enumerate(psd_rows):
            value = values[ch_idx] if ch_idx < len(values) else 0.0
            bar_y = psd_top + idx * (PSD_BAR_HEIGHT + PSD_BAR_GAP)
            label_surface = label_font.render(disp_name, True, color)
            screen.blit(label_surface, (cell_left, bar_y - 2))
            bar_x = psd_bar_start
            width = int(round((value / psd_max) * psd_bar_max_width)) if psd_max > 0 else 0
            width = max(1, width)
            bar_rect = pygame.Rect(bar_x, int(bar_y), width, PSD_BAR_HEIGHT)
            pygame.draw.rect(screen, color, bar_rect)
            pygame.draw.rect(screen, (30, 30, 50), bar_rect, 1)
            value_surface = value_font.render(f"{value:.0f}", True, BAR_VALUE_COLOR)
            value_x = min(cell_left + int(col_width) - value_surface.get_width() - 4, bar_rect.right + 4)
            value_y = max(cell_top, bar_rect.top - 2)
            screen.blit(value_surface, (value_x, value_y))

        if col == 0:
            separator_x = int(rect.left + inner_margin + col_width)
            pygame.draw.line(
                screen,
                (40, 40, 70),
                (separator_x, cell_top),
                (separator_x, cell_top + row_height - 8),
                1,
            )


def main() -> None:
    args = parse_args()
    inlet = connect_lsl(args.stream_name)
    screen, clock = init_pygame()

    buffer_len = int(args.buffer_seconds * cfg.SAMPLE_RATE_HZ)
    channel_buffers: List[Deque[float]] = [deque(maxlen=buffer_len) for _ in CHANNEL_INDICES]

    window_width, window_height = screen.get_size()
    grid_rows = max(PSD_GRID_ROWS, (len(CHANNEL_LABELS) + PSD_GRID_COLS - 1) // PSD_GRID_COLS)
    band_target = max(220, grid_rows * PSD_ROW_HEIGHT + 60)
    top_height = max(260, int(window_height * 0.42))
    total_needed = top_height + band_target + SLIDER_AREA_HEIGHT + 3 * PADDING + 10
    if total_needed > window_height:
        shrink = min(top_height - 160, total_needed - window_height)
        top_height -= max(0, shrink)
        total_needed = top_height + band_target + SLIDER_AREA_HEIGHT + 3 * PADDING + 10
        if total_needed > window_height:
            band_target = max(160, band_target - (total_needed - window_height))

    available_width = max(200, window_width - (Y_SLIDER_WIDTH + Y_SLIDER_GAP) - 3 * PADDING)
    min_panel_width = max(140, min(240, available_width // 2))
    if 2 * min_panel_width > available_width:
        min_panel_width = max(100, available_width // 2)
    time_width = max(min_panel_width, int(available_width * 0.5))
    freq_width = available_width - time_width
    if freq_width < min_panel_width:
        freq_width = min_panel_width
        time_width = available_width - freq_width
    if time_width < min_panel_width:
        time_width = min_panel_width
        freq_width = available_width - time_width

    time_rect = pygame.Rect(PADDING, PADDING, time_width, top_height)
    y_slider_rect = pygame.Rect(time_rect.right + Y_SLIDER_GAP, time_rect.top, Y_SLIDER_WIDTH, top_height)
    freq_left = y_slider_rect.right + PADDING
    freq_rect = pygame.Rect(freq_left, PADDING, freq_width, top_height)
    slider_rect = pygame.Rect(PADDING, time_rect.bottom + 10, window_width - 2 * PADDING, SLIDER_AREA_HEIGHT)
    band_top = slider_rect.bottom + PADDING
    available_band = max(0, window_height - band_top - PADDING)
    band_height = min(band_target, available_band)
    if band_height < min(available_band, 160):
        band_height = min(available_band, max(160, band_height))
    band_rect = pygame.Rect(PADDING, band_top, window_width - 2 * PADDING, band_height)

    x_zoom = 1.0
    y_zoom = 1.0
    dragging: str | None = None
    knob_rects: Dict[str, pygame.Rect] = {}
    bar_rects: Dict[str, pygame.Rect] = {}

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise SystemExit
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    raise SystemExit
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    pos = event.pos
                    for name, krect in knob_rects.items():
                        if krect.collidepoint(pos):
                            dragging = name
                            break
                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    dragging = None
                if event.type == pygame.MOUSEMOTION and dragging:
                    if dragging == "x" and "x" in bar_rects:
                        x_zoom = slider_pos_to_value(event.pos[0], X_ZOOM_RANGE[0], X_ZOOM_RANGE[1], bar_rects["x"])
                    elif dragging == "y" and "y" in bar_rects:
                        y_zoom = slider_pos_to_value_vertical(event.pos[1], Y_ZOOM_RANGE[0], Y_ZOOM_RANGE[1], bar_rects["y"])
            chunk, _ = inlet.pull_chunk(timeout=0.0, max_samples=int(cfg.SAMPLE_RATE_HZ * cfg.CHUNK_LENGTH_SEC))
            if chunk:
                for sample in chunk:
                    for idx, ch_idx in enumerate(CHANNEL_INDICES):
                        if ch_idx < len(sample):
                            channel_buffers[idx].append(sample[ch_idx])

            max_len = max((len(buf) for buf in channel_buffers), default=0)
            data = np.zeros((len(CHANNEL_INDICES), max_len), dtype=float)
            for idx, buf in enumerate(channel_buffers):
                if len(buf):
                    data[idx, -len(buf) :] = np.fromiter(buf, dtype=float)
            screen.fill(cfg.BACKGROUND_COLOR)
            draw_time_domain(screen, time_rect, data, x_zoom, y_zoom)

            # Draw sliders beneath time-domain plot
            knob_rects = {}
            bar_rects = {}
            knob_rects["x"], bar_rects["x"] = draw_slider(
                screen,
                pygame.Rect(slider_rect.left, slider_rect.top + 10, slider_rect.width, 28),
                "Time X",
                x_zoom,
                X_ZOOM_RANGE[0],
                X_ZOOM_RANGE[1],
                dragging == "x",
            )
            knob_rects["y"], bar_rects["y"] = draw_vertical_slider(
                screen,
                pygame.Rect(y_slider_rect.left, y_slider_rect.top, y_slider_rect.width, y_slider_rect.height),
                "Amplitude Y",
                y_zoom,
                Y_ZOOM_RANGE[0],
                Y_ZOOM_RANGE[1],
                dragging == "y",
            )

            if data.shape[1] > 0:
                freqs, spectrum = compute_fft(data, cfg.SAMPLE_RATE_HZ, args.max_freq)
            else:
                freqs, spectrum = np.array([]), np.array([])
            current_spectrum = spectrum if spectrum.size else np.zeros((len(CHANNEL_INDICES), 1))
            draw_frequency_domain(screen, freq_rect, freqs, current_spectrum)
            band_power = band_powers(freqs, current_spectrum)
            mu_beta_power = band_powers(freqs, current_spectrum, MU_BETA_BANDS)
            draw_band_panels(screen, band_rect, band_power, mu_beta_power)

            pygame.display.flip()
            clock.tick(cfg.REFRESH_RATE_HZ)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
