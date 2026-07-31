# Liquid-Lens Focus Calibration Tool

Builds a `z → diopter` lookup table for an Optotune liquid lens, using a
multi-camera Basler rig to measure the true `z` of static AprilTag targets.

## Hardware

| Device | Role |
|--------|------|
| 6× Basler cameras | Triangulate AprilTag `z` position |
| XIMEA CB160CG-LX-X8G3 | Measure focus sharpness through the liquid lens |
| Optotune liquid lens | Swept in focal-power (diopter) mode with temperature compensation |

Calibration data: `/home/nfc/braid-configs/calibration_charuco.xml`

To generate an AprilTag sheet: [shiqiliu-67.github.io/apriltag-generator](https://shiqiliu-67.github.io/apriltag-generator/)

## Installation

```bash
uv sync
```

The `optotune-lens` package is sourced from `../optotune-lens` (local path).
Basler Pylon SDK and XIMEA xiAPI runtime must be installed system-wide.

## Getting Started

The liquid lens focuses at a different real-world distance for every diopter
value you command it to. This tool builds the lookup table between the two:
it places an AprilTag at a known distance (measured by triangulating it with
the six Basler cameras), sweeps the lens through its diopter range while
watching a separate XIMEA camera behind the lens, and records the diopter at
which that tag is sharpest. Do this at enough distances and a `z → diopter`
curve is fit at the end — that curve is what a real application uses at
runtime to autofocus.

To run a calibration session:

1. Place an AprilTag (family `36h11`) somewhere in view of the Basler rig.
2. Start the tool: `uv run lens-calibrate`.
3. Press **SPACE** in the preview window — it triangulates the tag, sweeps
   the lens, and records the best-focus diopter for that position.
4. Move the tag to a new distance and press **SPACE** again. Repeat for
   ~10–15 positions spanning the range you care about.
5. Press **Q** to quit. This fits the `z → diopter` curve and writes
   a timestamped CSV of every raw measurement.

See [Procedure](#procedure) below for the full session walkthrough, and
[Output](#output) for what's in the CSV.

## Usage

```bash
uv run lens-calibrate [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--calibration` | `/home/nfc/braid-configs/calibration_charuco.xml` | Braid multi-camera calibration XML. If the default doesn't exist and no `--calibration` is given, the tool errors out rather than silently proceeding |
| `--port` | auto | Optotune lens controller serial port. Auto-detected by probing both `/dev/optotune_icc1c` and `/dev/optotune_ld` (each tried as both ICC-1C and Driver 4); set explicitly to restrict detection to one port |
| `--exposure` | `10000` | XIMEA exposure in µs |
| `--coarse-steps` | `20` | Steps in the full-range diopter sweep |
| `--fine-steps` | `40` | Steps in the narrow fine sweep per tag |
| `--fine-repeats` | `1` | Repeats per fine-sweep direction (hi→lo / lo→hi), for hysteresis stats |
| `--settle-ms` | `25` | Wait after each diopter change (ms). Minimum 25 ms — the lens driver is open-loop and needs that long to physically settle; lower values are rejected. Default is the fastest safe value |
| `--z-thresh` | `0.02` | Max z-spread (m) to treat tags as coplanar |
| `--tag-family` | `36h11` | AprilTag family (`36h11`, `25h9`, …) |
| `--debug` | off | Save focus-curve plots and ROI crops to `./debug/` after each sweep |

## Procedure

### Target setup

Use one or more AprilTag (family `36h11`) markers. Two modes are
auto-detected based on the spread of triangulated z values:

**Single-height mode** — all tags within `--z-thresh` of each other:
- Z values are fused (weighted by number of cameras per tag).
- One `(z, diopter)` data point is recorded per press of SPACE.
- Move the target to a new height and repeat (~10–15 positions for a good fit).

**Multi-height mode** — tags span more than `--z-thresh` in z:
- Each tag is triangulated and focused independently.
- One `(z, diopter)` data point per tag per press of SPACE.
- Useful for a stacked multi-plane target: get N points in one interaction.

### Session walkthrough

1. Place the target(s) below the cameras.
2. Run `uv run lens-calibrate`.
3. Press **SPACE** to measure, or **Q** to quit.
   - The Basler rig triangulates all visible tags.
   - The lens performs a coarse sweep (full diopter range) while the XIMEA
     camera auto-detects tags and builds per-tag focus ROIs.
   - For each tag, a fine sweep refines the peak using Gaussian log-space
     interpolation (Bonatti 2024, eq. 3.9), swept both hi→lo and lo→hi to
     characterize hysteresis (a warning is printed if the two directions
     disagree by more than 0.1 D).
4. Repeat for as many positions as needed.
5. On quit, a quadratic polynomial `D = a·z² + b·z + c` is fitted (found to
   fit as well as the vergence model over the short z ranges involved, and
   simpler to invert) and a timestamped CSV is written.

### Output

Two files are written:

- `calibrations/lens_calib_YYYYMMDD_HHMMSS.csv` (relative to the current
  directory) — full data. Columns: `z, diopter, sweep_direction, x, y,
  n_cameras, n_tags, focus_metric_peak, timestamp`. Each measurement
  contributes two rows per tag — one per sweep direction (`hi2lo` / `lo2hi`)
  — so the fit sees both branches of the hysteresis loop.
- `/home/nfc/src/OptoFly/calibrations/liquid_lens.csv` — just `z, dpt` (the
  `diopter` column renamed), which is exactly where OptoFly's
  `liquid_lens.calibration_file` expects to find it. If that file already
  exists, you're prompted `replace it with this calibration? [Y/n]` —
  defaults to yes, backing up the old file in place as
  `liquid_lens.csv.bak-YYYYMMDD_HHMMSS` before writing the new one.
  Declining instead saves the new calibration as
  `calibrations/lens_calib_YYYYMMDD_HHMMSS_optofly.csv`, alongside the full
  CSV, and leaves the existing `liquid_lens.csv` untouched.

To interpolate diopter from `z` in real time, load the full CSV and either
re-fit the polynomial (or the vergence model, if the runtime range grows
large enough for its curvature to matter) or use direct interpolation
(e.g. `numpy.interp`).

## Algorithm

- **Focus metric**: Tenengrad (mean squared Sobel gradient magnitude) — robust against bokeh artifacts on coarse targets like AprilTags.
- **Peak finding**: Gaussian log-space 3-point interpolation for sub-step precision; parabola fit and argmax as fallbacks.
- **Coarse sweep**: full diopter range, per-tag ROIs derived automatically from AprilTag detections in the XIMEA frame.
- **Fine sweep**: ±3 coarse steps around each tag's coarse peak, swept hi→lo then lo→hi (optionally repeated via `--fine-repeats`) to measure hysteresis.
- **Triangulation**: DLT with SVD using the 3×4 world→pixel projection matrices from the braid calibration XML; pixels are undistorted with OpenCV before triangulation.

## Module layout

```
src/liquid_lens_calibration/
├── main.py            CLI loop, CSV output, polynomial fit
├── calibration_io.py  Parse braid XML → per-camera intrinsics + projection matrices
├── cameras.py         Basler camera discovery, frame grab, buffer flush
├── focus_camera.py    XIMEA camera (gain 0, user exposure, full/ROI frame grab)
├── triangulate.py     AprilTag detection, undistortion, DLT triangulation
├── focus.py           Focus metric, coarse+fine sweep, peak interpolation
└── lens.py            Optotune lens wrapper (focal-power mode, diopter sweep)
```
