# Motor Imagery BCI Toolkit

A complete end-to-end pipeline for four-class motor imagery (MI) experiments built around OpenBCI + Lab Streaming Layer. The toolkit covers data acquisition, model training (Filter Bank CSP + SVM), offline evaluation, real-time cursor control, a visual debugging HUD, and a playful maze mini-game powered by your brain signals.

---

## Features at a Glance
- Guided acquisition UI with cue imagery, pause/resume, and baseline/rest collection ([data_acquisition.py](data_acquisition.py)).
- Training script that stitches multiple MAT files and exports reproducible model artifacts ([train_model.py](train_model.py)).
- Real-time demos: classic cursor control ([realtime_control.py](realtime_control.py)), fullscreen EEG visualizer ([realtime_visualizer.py](realtime_visualizer.py)), and the Xiaohui maze runner ([mi_maze_game.py](mi_maze_game.py)).
- Quickstart orchestrator that chains acquisition → training → control with a single command ([quickstart.py](quickstart.py)).
- Optional benchmarking on BCI Competition IV 2a for comparison against open datasets ([evaluate_bci2a.py](evaluate_bci2a.py)).

---

## Repository Layout
- [config/mi_config.py](config/mi_config.py): Central place for sampling rates, LSL names, UI colors, and the default electrode order (C3, C4, Cz, FC1, FC2, T3, T4, Fz).
- [algorithms/](algorithms): Filter Bank CSP implementation and SVM wrapper used everywhere.
- [data/](data): MAT files produced by acquisition (ignored if you point elsewhere).
- [models/](models): Saved model directories with `config.json`, `model.joblib`, and `metrics.json`.
- [res/](res): Optional sprites and fonts (maze hero image, Chinese-capable fonts, cue images, etc.).

---

## Prerequisites
1. **Python**: 3.9+ (tested on Anaconda Python 3.10).
2. **Packages** (install into your env):
   ```bash
   pip install numpy scipy scikit-learn mne pygame pylsl matplotlib joblib
   ```
   - `mne` is only needed for the BCICIV-2a evaluator.
   - Install OpenBCI Hub / LSL bridge separately so the EEG stream is published as `obci_eeg1` (or update the name in [config/mi_config.py](config/mi_config.py)).
3. **Hardware**: 8-channel OpenBCI montage wired to the new order (C3, C4, Cz, FC1, FC2, T3, T4, Fz) so the first eight stream channels align with the code.
4. **Assets** (optional): Drop custom cue PNGs into [res/](res) and fonts into [res/fonts](res/fonts) if you need localized overlays.

---

## Setup Checklist
1. Clone or download the repo.
2. Create/activate your Python environment.
3. Install dependencies (see above) and verify `python -m pylsl.examples.SendData` can publish to LSL.
4. (Optional) Edit [config/mi_config.py](config/mi_config.py) to adjust window sizes, class labels, or thresholds.

---

## 1. Acquire Training Data
1. Make sure the LSL EEG stream is live (`obci_eeg1` by default).
2. Run:
   ```bash
   python data_acquisition.py <subject_id> <session_id> <trials_per_class>
   ```
   - Example: `python data_acquisition.py 2 2 20` collects 20 trials/class plus rest.
   - Space bar toggles pause/resume, and the session will not start until you press space on the splash screen.
3. Output MAT files land in [data/](data) with the template `subject_<id>_session_<id>.mat`, storing channels, labels, timestamps, and metadata.

### Tips
- Use the rest override `--rest-epochs N` if you want extra REST segments.
- Cue images are resolved relative to [res/](res); missing images degrade gracefully with text prompts.

---

## 2. Train the Decoder
1. Point the training script at one or more MAT files:
   ```bash
   python train_model.py data/subject_2_session_2.mat
   ```
2. The script runs cross-validation (when enough trials exist), then retrains on the full dataset and saves artifacts under [models/](models) using the timestamp tag (you can override with `--output-dir` / `--tag`).
3. Contents of the model directory:
   - `config.json`: FBCSP settings + channel metadata.
   - `model.joblib`: Serialized FilterBankCSPClassifier.
   - `metrics.json`: Fold accuracies, training score, timestamps, and provenance.

---

## 3. Real-Time Demos
### Cursor Control UI
```bash
python realtime_control.py models/fbcsp_svm_model_<timestamp>/
```
- Keys: ESC closes the window; everything else is hands-free.
- UI shows the majority-voted command and confidence bar; cursor velocity is smoothed via damping and re-centered if it hits the bounds.

### EEG Visualizer
```bash
python realtime_visualizer.py [--stream-name obci_eeg1]
```
- Opens fullscreen; ESC exits.
- Upper left: time-domain traces; upper right: frequency PSD with band overlays.
- Bottom: per-channel band power bars plus Mu/Beta PSD summaries.
- Sliders: horizontal slider adjusts time zoom, vertical slider adjusts amplitude scaling in real time.

### Maze Mini-Game
```bash
python mi_maze_game.py models/fbcsp_svm_model_<timestamp>/
```
Key CLI flags:
- `--single-step` / `--continuous-step`: require a fresh command vs. allow holding the same command.
- `--maze-cols/--maze-rows`: choose maze size (odd numbers give symmetric carving).
- `--maze-open-factor`: remove extra walls to ease difficulty.

Flow:
1. Press space to spawn a maze.
2. Majority-voted MI commands move Xiaohui one cell at a time; reach the green EXIT to win.
3. Space again restarts with a new maze.

---

## 4. Quickstart Pipeline
If you just want to record → train → test in one go:
```bash
python quickstart.py
```
Flags let you skip individual stages:
- `--skip-acquire`, `--skip-train`, `--skip-control`.
- `--data-dir` and `--models-dir` change where MAT files and models are read/written.

---

## 5. Evaluate on BCI Competition IV 2a
Use this to benchmark against the public dataset (requires `.gdf` files and the official true-label `.mat` archives):
```bash
python evaluate_bci2a.py --dataset-dir <path_to_gdf> --label-dir <path_to_labels>
```
- `--subjects` selects which A01–A09 subjects to run.
- `--use-all-channels` or `--channel-indices` lets you experiment beyond the default eight.
- Prints per-fold accuracy and confusion matrices; no artifacts are saved.

---

## Troubleshooting & FAQ
- **LSL stream not found**: Ensure the OpenBCI GUI or Hub publishes the same `LSL_STREAM_NAME` as configured. Override via `--stream-name` on any runtime script.
- **Wrong electrode order**: The code assumes the first eight LSL channels map to `(C3, C4, Cz, FC1, FC2, T3, T4, Fz)`. Rewire or edit `CHANNEL_LABELS` plus dependent indices (e.g., `CHANNEL_INDICES` in [realtime_visualizer.py](realtime_visualizer.py)).
- **Fonts render garbled Chinese text**: Drop Noto Sans SC (OTF/TTF) into [res/fonts](res/fonts). The maze game and overlays try those paths first.
- **Model not loading**: Confirm the directory contains `config.json`/`model.joblib`. The load helpers expect the exact structure produced by [train_model.py](train_model.py).
- **Performance tuning**: Adjust `SLIDING_WINDOW_SEC`, `WINDOW_STEP_SEC`, and `CONFIDENCE_THRESHOLD` inside [config/mi_config.py](config/mi_config.py). For example, reducing $\tau$ (window) increases responsiveness but may cut accuracy.

---

## Contributing
Issues and pull requests are welcome. Please run black/ruff (or your preferred formatter/linter) on any Python edits and include a short description of hardware/software context when reporting bugs.

Happy hacking, and may your MI signals guide Xiaohui safely through every maze! 🎮🧠
