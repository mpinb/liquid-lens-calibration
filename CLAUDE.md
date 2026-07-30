# Liquid-Lens Focus Calibration Tool

A command-line tool that builds a `z → diopter` lookup table for an Optotune
liquid lens, using a multi-camera Basler rig to measure the true `z` of one or
more static AprilTag targets.

This file documents the **as-built** tool (originally a build spec — the
implementation has since evolved past several of the original decisions; the
sections below reflect current behavior, not the initial plan). Keep the code
clear and readable — a user should be able to follow it end to end. Do not
over-engineer. Plain functions and a small number of modules are fine; no
plugin systems, no abstract base classes, no config frameworks.

## What it does

1. User places one or more AprilTag markers below the cameras and presses
   Enter (in the live XIMEA preview window).
2. Tool grabs images from all available Basler cameras (`pypylon`), detects
   every visible tag, and triangulates each tag's xyz using the precomputed
   multi-camera calibration.
3. Sweeps the liquid lens through its full diopter range once (coarse pass),
   auto-detecting a focus ROI **per tag** from AprilTag corners seen in the
   XIMEA frame — no manual ROI selection is needed per step.
4. For each detected tag, sweeps a narrow window around its coarse peak in
   both directions (hi→lo, then lo→hi) to measure hysteresis, and resolves a
   sub-step best-focus diopter via Gaussian log-space interpolation (parabola
   fit and argmax are fallbacks).
5. Reports per-step statistics, prompts the user to move the target(s),
   repeats.
6. On quit, fits a quadratic `z → diopter` polynomial and writes a timestamped
   CSV of all raw data points.

## Inputs / hardware

- **Multi-camera calibration:** `/home/nfc/braid-configs/calibration_charuco.xml`
  (braid format), and the `--calibration` default — no need to pass it
  explicitly on this rig. If that default path doesn't exist and the user
  hasn't passed `--calibration`, `main.py` errors out at arg-parsing time
  (via `parser.error`) instead of silently proceeding; an explicit
  `--calibration <path>` bypasses that check entirely, even if that path is
  also missing (parsing it then just fails normally). Each
  `<single_camera_calibration>` has a 3×4 `<calibration_matrix>` (linear
  pinhole projection, world→pixel homogeneous) plus `<non_linear_parameters>`
  (fc1/fc2, cc1/cc2, k1/k2/k3, p1/p2 — standard OpenCV-style intrinsics +
  distortion). Parsed for all `n` cameras by `cam_id` in `calibration_io.py`.
- **Cameras:** Basler, opened via `pypylon`. Matched to a `cam_id` in the XML
  by serial number — `cam_id` is `Basler-<serial>` (`cameras.py`).
- **Liquid lens controller:** the `optotune-lens` package
  (`/home/nfc/src/optotune-lens`, installed as a `uv` path dependency), which
  supports two controller models with incompatible wire protocols: the
  original **Lens Driver 4** (`Lens` class, binary protocol) and the newer
  **ICC-1C** (`ICC1C` class, ASCII "Simple Mode" protocol). `lens.py`'s
  `open_lens()` **auto-detects which one is on `--port`** — it tries
  connecting as each class in turn (`_detect_driver`), and each class's
  `__init__` already performs its own handshake and raises `LensError` if the
  device on the other end doesn't answer as expected, so identification just
  means catching that and trying the next class. If neither responds,
  `open_lens` raises a combined `LensError` and `main.py` prints it and exits
  (rather than a raw traceback). Whichever class is detected, both expose the
  same `set_diopter()` used everywhere else in the tool, so nothing
  downstream of `open_lens` needs to know which controller is connected.
  Both are driven in **focal-power (diopter) mode**: for the `Lens` (Driver
  4), `lens.py` calls `lens.to_focal_power_mode()` unless already in that
  mode (skips the mode-switch to avoid a spurious "already in this mode"
  firmware error) and reads the range off `.min_diopter`/`.max_diopter`; for
  `ICC1C`, there's no mode concept — `to_focal_power_mode()` just validates
  the connected lens has EEPROM calibration data, and the range comes from
  `get_diopter_min()`/`get_diopter_max()`. Temperature compensation and
  diopter range otherwise come from the package itself.
  **The driver is open-loop** (no internal position feedback) — per Optotune,
  commanding a new diopter faster than ~25 ms doesn't let the lens finish
  settling, so a frame grabbed right after is measuring a transient, not the
  commanded focal power. `main.py` enforces `--settle-ms >= 25` (`_MIN_SETTLE_MS`)
  for this reason; don't lower it below that floor even for faster sweeps.
  `--settle-ms` defaults to `_MIN_SETTLE_MS` (25) — the fastest sweep the
  lens can settle at.
  The ICC-1C reportedly halves this settling time and adds a "smart step"
  mode (Pro Mode, not implemented by `ICC1C` — see its docstring) — worth
  revisiting `_MIN_SETTLE_MS` if sweep speed becomes a bottleneck now that
  ICC-1C is supported.
- **Focus camera:** a **separate** XIMEA CB160CG-LX-X8G3 sitting behind the
  liquid lens. This is the camera the focus metric is computed on — it is NOT
  one of the six Basler triangulation cameras. Controlled via `ximea-py`
  (standard xiAPI) in `focus_camera.py`:
  - **Gain is hardcoded to 0.0** — not exposed as an option.
  - **Exposure is user-settable** via `--exposure` (default `10000` µs).
  - **Newest-frame guarantee:** `set_buffers_queue_size(2)` is attempted at
    open time; if that fails, every grab falls back to flushing 3 frames
    first. Either way `grab_full_frame()` / `grab_roi_frame()` always return
    the current frame, never a stale buffered one.
  - **ROI is no longer a single manual selection.** `XimeaFocusCamera.select_roi()`
    (interactive `cv2`-based drag-box) still exists but is currently unused by
    `main.py`. Instead, **per-tag ROIs are auto-detected every sweep** from
    AprilTag corner detections in the XIMEA frame during the coarse pass (see
    `focus.py::sweep_all_tags`) — this supports multiple simultaneous tags and
    removes the need to redraw a box each session. If you reintroduce manual
    ROI selection, update this section.

## Triangulation (`triangulate.py`)

For each camera that sees a tag:
1. Detect all AprilTags (OpenCV `cv2.aruco`, family selectable via
   `--tag-family`, default `36h11`), take each tag's center pixel (mean of its
   4 corners).
2. Undistort the pixel using that camera's intrinsics + distortion
   (`cv2.undistortPoints` with `P=K`, i.e. stays in pixel coordinates — braid
   convention: `calibration_matrix` maps world→pixel, so undistortion must
   also stay in pixel space).
3. Linear DLT triangulation across all cameras that saw that tag ID, using the
   3×4 `calibration_matrix` rows (build the `A` matrix, solve by SVD,
   dehomogenize). Tags seen by fewer than 2 calibrated cameras are dropped.

Every visible tag ID is triangulated independently (`detect_and_triangulate`
returns `{tag_id: (x, y, z, n_cameras)}`). `main.py` then decides, per
measurement, whether the tags are coplanar (see below).

## Per-step procedure (`main.py` loop)

1. Live XIMEA preview window; Enter = measure, Q/Esc = quit (keys read from
   the OpenCV window, not the terminal).
2. Flush stale buffers on all six Baslers and the XIMEA, grab one frame per
   Basler, detect + triangulate every visible tag.
3. **Coplanar vs multi-height decision:** if the z-spread across all
   triangulated tags is below `--z-thresh` (default 2 cm), treat them as one
   physical height and fuse `x/y/z` (weighted by camera count per tag) into a
   single data point later. Otherwise treat each tag as an independent height.
4. Coarse sweep: full diopter range, `--coarse-steps` steps. At each step,
   detect AprilTags in the XIMEA frame (CLAHE-enhanced, permissive detector
   params for defocus tolerance) and track a bounding-box ROI per tag ID.
   Frames are retained so each tag's full coarse focus curve can be computed
   retroactively once its final ROI is known (tags are often undetectable
   except near their own focus point).
5. Fine sweep per tag: a window of `±3` coarse steps around that tag's coarse
   peak, swept hi→lo then lo→hi (`--fine-steps` steps each, `--fine-repeats`
   repeats per direction) to characterize hysteresis. Each direction's peak is
   resolved via `_find_peak` (Gaussian log-space interpolation → parabola fit
   → argmax cascade); the reported best-focus diopter is the mean of the two
   directions' peaks, and a warning is printed if hysteresis exceeds
   `HYSTERESIS_THRESH` (0.1 D).
6. Two CSV rows are recorded per tag per measurement — one per sweep direction
   (`sweep_direction: hi2lo` / `lo2hi`) — so the fit sees both branches of the
   hysteresis loop rather than an average-only point.
7. `--debug` additionally saves a focus-curve plot and ROI crop per tag to
   `./debug/`.
8. Prompt again: move the target(s), press Enter to measure, or Q to quit.

## Output

On quit, writes two files — but only if at least 4 data points were
collected (i.e. two full measurements × two sweep directions). With fewer,
`main.py` prints a message and exits without fitting or writing anything.

`calibrations/lens_calib_YYYYMMDD_HHMMSS.csv`
(relative to cwd, `_LOCAL_CALIB_DIR`, created if missing) holds the full
data:

```
z, diopter, sweep_direction, x, y, n_cameras, n_tags, focus_metric_peak, timestamp
```

The second is a minimal OptoFly export — just `z, dpt` (the `diopter` column
renamed, all rows kept — both sweep directions) — written directly to
`_OPTOFLY_CALIB_PATH` (`/home/nfc/src/OptoFly/calibrations/liquid_lens.csv`),
which is exactly where OptoFly's `setup_lens_calibration`
(`src/processes/lens.py`) expects `liquid_lens.calibration_file` to point.
This path is intentionally independent of `_LOCAL_CALIB_DIR` — it's OptoFly's
own calibrations folder, not this tool's local output folder, and the two
happen to share a directory name by coincidence. If `_OPTOFLY_CALIB_PATH`
already exists, `_write_csv` prompts on stdin before overwriting; declining
falls back to writing `lens_calib_YYYYMMDD_HHMMSS_optofly.csv` under
`_LOCAL_CALIB_DIR` instead, so no calibration run is ever silently lost.
This prompt runs after the cv2 preview windows are torn down, so it's safe
to use a blocking terminal `input()` here.

`main.py` then fits **`D = a·z² + b·z + c`** (quadratic polynomial via
`np.polyfit`) and prints the coefficients + residual RMS. This was found to
fit as well as the theoretical vergence model over the short z ranges
involved, and is simpler to invert (`np.polyval`). See `CONTEXT.md` for the
original vergence-model (`D = a/(z - z0) + b`) rationale — that model can
still be refit from the raw CSV if the runtime range grows large enough for
its curvature to matter (e.g. via `scipy.optimize.curve_fit`).

## Module layout

```
src/liquid_lens_calibration/
├── main.py            CLI loop, live preview, coplanar/multi-height logic, CSV output, polynomial fit
├── calibration_io.py  Parse braid XML → per-camera intrinsics + projection matrices
├── cameras.py         Basler discovery (by serial), frame grab, buffer flush
├── focus_camera.py    XIMEA camera (gain 0, user exposure, newest-frame grab, ROI crop, unused manual select_roi)
├── triangulate.py     AprilTag detection (cv2.aruco), undistortion, DLT triangulation
├── focus.py           Focus metric (Tenengrad), auto per-tag ROI tracking, coarse+fine sweep, peak interpolation, debug plots
└── lens.py            Optotune lens wrapper (focal-power mode, diopter range)
```

Use type hints and docstrings throughout (project convention).

## Resolved implementation decisions

These were open questions in the original build spec; recorded here so they
aren't re-litigated or re-guessed by a future session:

1. **`optotune-lens` API** — `Lens(port)` (Driver 4: `.mode`
   (`OperatingMode` enum), `.to_focal_power_mode()`, `.min_diopter` /
   `.max_diopter`) and `ICC1C(port)` (Simple Mode: no `.mode`,
   `.to_focal_power_mode()` just validates, `.get_diopter_min()` /
   `.get_diopter_max()`). `open_lens()` auto-detects which is connected. See
   `lens.py`.
2. **AprilTag detection** — OpenCV's built-in `cv2.aruco` (not `pupil-apriltags`
   or the standalone `apriltag` package). Family is selectable via
   `--tag-family` (`TAG_FAMILIES` dict in `triangulate.py`), default `36h11`.
3. **`cam_id` ↔ pypylon serial** — confirmed `Basler-<serial>` format; matched
   in `discover_basler_cameras`.
4. **XIMEA exposure default** — `10000` µs, user-overridable via `--exposure`.
   ROI selection is no longer interactive per the design above (see Focus
   camera section) — `--debug` mode's saved ROI crops are the practical way
   to sanity-check what's being measured.

## Installing and Using Claude Code

Claude Code is an AI-powered coding assistant that runs in your terminal.

### Install

Pick one:

```bash
# macOS, Linux, or WSL
curl -fsSL https://claude.ai/install.sh | bash

# npm (any platform, requires Node.js 22+)
npm install -g @anthropic-ai/claude-code

# Homebrew (macOS)
brew install --cask claude-code
```

Windows PowerShell: `irm https://claude.ai/install.ps1 | iex`

Verify: `claude --version` should print a version like `2.1.X (Claude Code)`.

### First run

```bash
cd ~/src/liquid_lens_calibration
claude
```

The first run opens a browser to authenticate (Claude Pro/Max, Team/Enterprise, Console, or an `ANTHROPIC_API_KEY` environment variable). Credentials are then stored locally — no repeat login.

If this repo's `CLAUDE.md` is ever missing, running `/init` inside a Claude Code session regenerates it from the current codebase.

### Useful commands

- `/help` — list all commands
- `/clear` — reset conversation history
- Shift+Tab — cycle permission mode (`plan` / default / `acceptEdits`)
