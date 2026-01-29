"""Shared configuration for the four-class motor imagery BCI pipeline."""
from pathlib import Path

# LSL / acquisition parameters
LSL_STREAM_NAME = "obci_eeg1"
LSL_STREAM_TYPE = "EEG"
CHANNEL_LABELS = [
    "C3",
    "C4",
    "Cz",
    "FC1",
    "FC2",
    "T3",
    "T4",
    "Fz",
]
EXPECTED_CHANNEL_COUNT = len(CHANNEL_LABELS)
SAMPLE_RATE_HZ = 250  # OpenBCI Cyton default
CHUNK_LENGTH_SEC = 0.125  # 250 ms chunks for lower latency

# Data collection / storage
DATA_ROOT = Path("data")
MAT_FILE_TEMPLATE = "subject_{subject_id}_session_{session_id}.mat"
EVENT_LABELS = {
    0: "REST",
    1: "UP",
    2: "DOWN",
    3: "LEFT",
    4: "RIGHT",
}
EVENT_CUES = {
    0: "Rest",
    1: "↑ Tongue",
    2: "↓ Foot",
    3: "← Left Hand",
    4: "→ Right Hand",
}
EPOCH_DURATION_SEC = 4.0
REST_EPOCH_DURATION_SEC = 4.0
PRE_EVENT_MARGIN_SEC = 1.5
INTER_TRIAL_INTERVAL_SEC = 3.0
BASELINE_DURATION_SEC = 6.0

# Filter bank CSP parameters
FILTER_BANKS = [
    (8, 10),
    (10, 16),
    (16, 24),
    (24, 32),

    #     (8, 12),
    # (12, 16),
    # (16, 24),
    # (24, 32),
]
CSP_COMPONENTS_PER_BAND = 2
SVM_KERNEL = "linear"
SVM_C = 1.0
SVM_CLASS_WEIGHT = "balanced"

# Training / evaluation
CROSS_VALIDATION_FOLDS = 5
MODEL_OUTPUT_DIR = Path("models")
MODEL_ARTIFACT_BASENAME = "fbcsp_svm_model"

# Real-time decoding
SLIDING_WINDOW_SEC = 2.0
WINDOW_STEP_SEC = 0.15
MAJORITY_VOTE_WINDOW = 1  # number of recent predictions to smooth commands
CONFIDENCE_THRESHOLD = 0.6  # require minimal SVM decision function magnitude
IDLE_COMMAND = "REST"

# UI parameters
SCREEN_SIZE = (1200, 900)
REFRESH_RATE_HZ = 30
FONT_NAME = "Arial"
BACKGROUND_COLOR = (10, 10, 40)
CURSOR_COLOR = (255, 200, 0)
RESOURCE_ROOT = Path("res")
CUE_IMAGE_PATHS = {
    1: RESOURCE_ROOT / "up.png",
    2: RESOURCE_ROOT / "down.png",
    3: RESOURCE_ROOT / "left.png",
    4: RESOURCE_ROOT / "right.png",
}
CURSOR_RADIUS = 10
CURSOR_SPEED_PX = 6.0
CURSOR_DAMPING = 0.55  # velocity decay per frame

