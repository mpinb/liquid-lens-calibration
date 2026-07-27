"""CLI entry point for liquid-lens focus calibration.

Usage::

    uv run python main.py [--exposure 10000] [--calibration <path>]
                          [--port /dev/optotune_ld]
                          [--coarse-steps 20] [--fine-steps 20]
                          [--z-thresh 0.02]
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from liquid_lens_calibration.calibration_io import parse_calibration_xml
from liquid_lens_calibration.cameras import discover_basler_cameras, grab_frame, flush_buffers as flush_basler
from liquid_lens_calibration.focus_camera import XimeaFocusCamera
from liquid_lens_calibration.triangulate import detect_and_triangulate, TAG_FAMILIES
from liquid_lens_calibration.focus import sweep_all_tags, show_preview, PREVIEW_WIN
from liquid_lens_calibration.lens import open_lens


def _live_wait(focus_cam: XimeaFocusCamera) -> str:
    """Show a live XIMEA preview and wait for a keypress in the preview window.

    Returns ``'measure'`` (Enter) or ``'quit'`` (Q / Escape).
    The terminal prompt is printed once; subsequent interaction is via the window.
    """
    print("  [preview window]  Enter = measure    Q = quit", flush=True)
    while True:
        frame = focus_cam.grab_full_frame()
        show_preview(frame, "Enter = measure    Q = quit")
        key = cv2.waitKey(30)
        if key in (13, 10):        # Enter
            return "measure"
        if key in (ord("q"), ord("Q"), 27):  # q / Q / Esc
            return "quit"


_CALIB_DEFAULT = Path("/home/nfc/braid-configs/calibration_charuco.xml")

# The Optotune driver is open-loop (no internal position feedback). Per
# Optotune, commanding a new focal power faster than ~25 ms means the lens
# hasn't finished settling from the previous command, so the frame grabbed
# at the "new" diopter is actually measuring a transient — silently
# corrupting the focus curve. This is a hardware floor, not a tunable knob.
_MIN_SETTLE_MS = 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a z → diopter lookup table for an Optotune liquid lens."
    )
    parser.add_argument(
        "--calibration",
        default=str(_CALIB_DEFAULT),
        help="Path to braid multi-camera calibration XML (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        default="/dev/optotune_ld",
        help="Serial port for the Optotune lens (default: %(default)s)",
    )
    parser.add_argument(
        "--exposure",
        type=int,
        default=10000,
        help="XIMEA exposure in microseconds (default: %(default)s)",
    )
    parser.add_argument(
        "--coarse-steps",
        type=int,
        default=20,
        help="Diopter steps in the coarse sweep (default: %(default)s)",
    )
    parser.add_argument(
        "--fine-steps",
        type=int,
        default=40,
        help="Diopter steps in the fine sweep per tag (default: %(default)s)",
    )
    parser.add_argument(
        "--fine-repeats",
        type=int,
        default=1,
        help="Number of times to repeat each fine sweep direction (default: %(default)s)",
    )
    parser.add_argument(
        "--settle-ms",
        type=int,
        default=_MIN_SETTLE_MS,
        help=(
            f"Milliseconds to wait after setting diopter (default: %(default)s). "
            f"Must be >= {_MIN_SETTLE_MS} ms — the lens is open-loop and needs "
            f"that long to physically settle; faster commands measure an "
            f"unsettled transient."
        ),
    )
    parser.add_argument(
        "--z-thresh",
        type=float,
        default=0.02,
        help=(
            "Max z-spread (metres) to treat all visible tags as coplanar "
            "and fuse their z values (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--tag-family",
        default="36h11",
        choices=list(TAG_FAMILIES),
        help="Marker dictionary to use for detection (default: %(default)s)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Save a focus-curve plot and ROI crop for each tag after every sweep. "
            "Files are written to ./debug/ as debug_tag<id>_<time>_curve.png "
            "and debug_tag<id>_<time>_roi.jpg."
        ),
    )
    args = parser.parse_args()
    if args.settle_ms < _MIN_SETTLE_MS:
        parser.error(
            f"--settle-ms {args.settle_ms} is below the lens's minimum settling "
            f"time ({_MIN_SETTLE_MS} ms); the driver is open-loop, so faster "
            f"commands would measure the lens mid-transient and corrupt the "
            f"calibration."
        )
    if args.calibration == str(_CALIB_DEFAULT) and not _CALIB_DEFAULT.exists():
        parser.error(
            f"Default calibration file not found: {_CALIB_DEFAULT}\n"
            f"Pass --calibration <path> to point at a different calibration XML."
        )
    return args


def main() -> None:
    args = parse_args()

    calibrations = parse_calibration_xml(args.calibration)
    print(f"Loaded {len(calibrations)} camera calibration(s) from {args.calibration}")

    basler_cameras = discover_basler_cameras(set(calibrations.keys()))
    print(f"Matched {len(basler_cameras)} Basler camera(s)")

    lens, (d_min, d_max) = open_lens(args.port)
    settle_s = args.settle_ms / 1000.0
    dataset: list[dict[str, float | int | str]] = []

    try:
        with XimeaFocusCamera(exposure_us=args.exposure) as focus_cam, lens:
            cv2.namedWindow(PREVIEW_WIN, cv2.WINDOW_NORMAL)
            print(f"Optotune lens diopter range: {d_min:.2f} to {d_max:.2f} D")
            print(f"Coarse steps: {args.coarse_steps}  Fine steps: {args.fine_steps}")
            print(f"Tags within {args.z_thresh * 1000:.0f} mm z-spread → coplanar (z fused)")
            if args.debug:
                print(f"Debug mode ON — plots and ROI crops saved to ./debug/")

            print("\n" + "=" * 60)
            print("Calibration loop")
            print("  Place AprilTag(s), then press Enter to measure.")
            print("  Multiple tags at different heights → one data point each.")
            print("  Multiple tags at similar height → fused z, one data point.")
            print("  Keys are read from the preview window (click it first).")
            print("=" * 60)

            while True:
                if _live_wait(focus_cam) == "quit":
                    break

                # Flush all camera buffers
                for cam in basler_cameras.values():
                    flush_basler(cam, n=3)
                focus_cam.flush(n=3)

                # Grab Basler frames and triangulate all visible tags
                frames: dict[str, np.ndarray] = {}
                for cam_id, cam in basler_cameras.items():
                    frames[cam_id] = grab_frame(cam)

                try:
                    tag_results = detect_and_triangulate(
                        frames, calibrations, tag_family=args.tag_family
                    )
                except RuntimeError as e:
                    print(f"  Triangulation failed: {e}")
                    continue

                print(f"  Basler detected {len(tag_results)} tag(s):")
                for tid, (x, y, z, n_cam) in sorted(tag_results.items()):
                    print(f"    tag {tid}: x={x:.4f}, y={y:.4f}, z={z:.4f} m  ({n_cam} cameras)")

                # Determine mode: coplanar (fuse z) or multi-height (per tag)
                zs = [v[2] for v in tag_results.values()]
                z_spread = max(zs) - min(zs)
                single_height = z_spread < args.z_thresh

                if single_height and len(tag_results) > 1:
                    print(f"  z-spread={z_spread * 1000:.1f} mm < {args.z_thresh * 1000:.0f} mm → coplanar, fusing z")

                # Sweep lens — per-tag ROIs auto-detected from XIMEA
                print(
                    f"  Sweeping: {args.coarse_steps} coarse + "
                    f"{args.fine_steps} fine steps × {args.fine_repeats} repeat(s) per tag …"
                )
                sweep_results = sweep_all_tags(
                    lens,
                    focus_cam,
                    args.tag_family,
                    (d_min, d_max),
                    n_coarse=args.coarse_steps,
                    n_fine=args.fine_steps,
                    n_fine_repeats=args.fine_repeats,
                    settle_s=settle_s,
                    debug=args.debug,
                    debug_dir=Path("debug"),
                )

                if not sweep_results:
                    print("  No tags detected in XIMEA during sweep — skipping.")
                    continue

                timestamp = datetime.now().isoformat(timespec="seconds")

                if single_height:
                    # Fuse z values weighted by number of cameras per tag
                    weights = np.array(
                        [tag_results[tid][3] for tid in tag_results], dtype=np.float64
                    )
                    all_x = [tag_results[tid][0] for tid in tag_results]
                    all_y = [tag_results[tid][1] for tid in tag_results]
                    all_z = [tag_results[tid][2] for tid in tag_results]
                    z_fused = float(np.average(all_z, weights=weights))
                    x_fused = float(np.average(all_x, weights=weights))
                    y_fused = float(np.average(all_y, weights=weights))
                    n_views = int(weights.sum())

                    # Pick the sweep result with the best (highest) metric peak
                    best_tid = max(sweep_results, key=lambda t: sweep_results[t][3])
                    best_d, best_d_hi2lo, best_d_lo2hi, peak_m, *_ = sweep_results[best_tid]

                    print(
                        f"  Best focus: {best_d:.3f} D  "
                        f"(↓{best_d_hi2lo:.3f} ↑{best_d_lo2hi:.3f}, "
                        f"z={z_fused:.4f} m, metric peak={peak_m:.1f}, "
                        f"{len(tag_results)} tag(s), {n_views} views)"
                    )
                    base = {
                        "z": z_fused,
                        "x": x_fused,
                        "y": y_fused,
                        "n_cameras": n_views,
                        "n_tags": len(tag_results),
                        "focus_metric_peak": peak_m,
                        "timestamp": timestamp,
                    }
                    dataset.append({**base, "diopter": best_d_hi2lo, "sweep_direction": "hi2lo"})
                    dataset.append({**base, "diopter": best_d_lo2hi, "sweep_direction": "lo2hi"})

                else:
                    # Multi-height: two rows per tag (one per sweep direction)
                    added = 0
                    for tid, (best_d, best_d_hi2lo, best_d_lo2hi, peak_m, *_) in sorted(
                        sweep_results.items()
                    ):
                        if tid not in tag_results:
                            print(f"  Tag {tid} seen in XIMEA but not triangulated — skipping.")
                            continue
                        x, y, z, n_cam = tag_results[tid]
                        print(
                            f"  Tag {tid}: z={z:.4f} m → avg {best_d:.3f} D  "
                            f"(↓{best_d_hi2lo:.3f} ↑{best_d_lo2hi:.3f}, "
                            f"metric peak={peak_m:.1f}, {n_cam} cameras)"
                        )
                        base = {
                            "z": z,
                            "x": x,
                            "y": y,
                            "n_cameras": n_cam,
                            "n_tags": 1,
                            "focus_metric_peak": peak_m,
                            "timestamp": timestamp,
                        }
                        dataset.append({**base, "diopter": best_d_hi2lo, "sweep_direction": "hi2lo"})
                        dataset.append({**base, "diopter": best_d_lo2hi, "sweep_direction": "lo2hi"})
                        added += 1
                    if added == 0:
                        print("  No matching tags between XIMEA and Basler — skipping.")
                        continue

                print(f"  Total data points collected: {len(dataset)}")

    finally:
        cv2.destroyAllWindows()
        for cam in basler_cameras.values():
            try:
                cam.Close()
            except Exception:
                pass

    if len(dataset) < 4:
        print(f"Only {len(dataset)} data point(s) — need at least 4 (2 measurements × 2 directions). No CSV written.")
        return

    z_vals = np.array([d["z"] for d in dataset], dtype=np.float64)
    d_vals = np.array([d["diopter"] for d in dataset], dtype=np.float64)

    print(f"\nFitting quadratic polynomial: D = a·z² + b·z + c")
    print(f"  {len(dataset)} data points  z=[{z_vals.min():.3f}, {z_vals.max():.3f}] m  "
          f"D=[{d_vals.min():.3f}, {d_vals.max():.3f}] diopter")

    _fit_polynomial(z_vals, d_vals)

    _write_csv(dataset)


def _fit_polynomial(z_vals: np.ndarray, d_vals: np.ndarray) -> None:
    """Fit D = a·z² + b·z + c and print results.

    A quadratic polynomial fits significantly better than the vergence model
    (D = a/(z-z0) + b) over short z ranges (< 0.5 m) where the hyperbolic
    curvature is indistinguishable from linear within measurement noise.
    Use ``np.polyval(coefs, z)`` for real-time lookup.
    """
    coefs = np.polyfit(z_vals, d_vals, 2)
    pred = np.polyval(coefs, z_vals)
    rms = float(np.sqrt(np.mean((d_vals - pred) ** 2)))
    a, b, c = coefs
    print(f"  D = {a:.4f}·z² + {b:.4f}·z + ({c:.4f})")
    print(f"  Residual RMS = {rms:.4f} D")
    print(f"  coefs (for np.polyval) = [{a:.6f}, {b:.6f}, {c:.6f}]")


def _write_csv(dataset: list[dict]) -> None:
    if not dataset:
        return
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"lens_calib_{timestamp_str}.csv"
    fieldnames = [
        "z", "diopter", "sweep_direction", "x", "y",
        "n_cameras", "n_tags", "focus_metric_peak", "timestamp",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)
    print(f"\nData saved to {csv_path}")

    optofly_path = f"lens_calib_{timestamp_str}_optofly.csv"
    with open(optofly_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["z", "dpt"])
        writer.writerows((d["z"], d["diopter"]) for d in dataset)
    print(f"OptoFly-compatible data saved to {optofly_path}")
    print("Done.")


if __name__ == "__main__":
    main()
