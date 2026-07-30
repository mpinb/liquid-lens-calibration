"""Focus metric computation and best-focus peak finding."""

import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
import cv2

from liquid_lens_calibration.triangulate import detect_apriltags, TAG_FAMILIES

# Matplotlib is imported lazily (only when debug=True) but the backend must be
# set before pyplot is ever loaded, so we do it at module import time.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class _Lens(Protocol):
    def set_diopter(self, value: float) -> None: ...


class _FocusCam(Protocol):
    def grab_full_frame(self) -> npt.NDArray[np.uint8]: ...
    def grab_roi_frame(self, roi: tuple[int, int, int, int]) -> npt.NDArray[np.uint8]: ...

PREVIEW_WIN = "XIMEA Live"
HYSTERESIS_THRESH = 0.1  # diopters — warn if the two fine-sweep directions disagree more than this


def show_preview(frame: npt.NDArray[np.uint8], text: str = "") -> None:
    """Update the live preview window with a half-resolution frame and text overlay.

    Call ``cv2.waitKey(1)`` after this to process GUI events.
    """
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    if small.ndim == 2:
        display = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    else:
        display = small.copy()
    if text:
        cv2.putText(display, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2, cv2.LINE_AA)
    cv2.imshow(PREVIEW_WIN, display)


def focus_metric(patch: npt.NDArray[np.uint8]) -> float:
    """Tenengrad focus metric (mean squared Sobel gradient magnitude).

    Responds to edge sharpness and is robust against out-of-focus bokeh
    artifacts that fool LoG on coarse targets like AprilTags. Higher values
    indicate sharper focus.

    Args:
        patch: Grayscale image patch.

    Returns:
        Scalar focus metric.
    """
    gx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
    return float((gx ** 2 + gy ** 2).mean())


def _gaussian_log_peak(
    d_prev: float,
    d_m: float,
    d_next: float,
    f_prev: float,
    f_m: float,
    f_next: float,
) -> float:
    """Sub-step peak diopter via Gaussian interpolation in log-space.

    Models the focus curve as F = F_p * exp(-½ ((d−d̄)/σ)²). With three
    uniformly-spaced samples the peak d̄ has a closed-form solution
    (Bonatti 2024, eq. 3.9).

    Args:
        d_prev, d_m, d_next: Uniformly-spaced diopter values. May be in either
            increasing (lo→hi sweep) or decreasing (hi→lo sweep) order — the
            formula is algebraically invariant to reversal.
        f_prev, f_m, f_next: Corresponding focus metrics (must all be > 0).

    Returns:
        Interpolated peak diopter, clamped to [min(d_prev,d_next), max(d_prev,d_next)].
        Falls back to d_m if the denominator is near zero.
    """
    ln_prev = np.log(max(f_prev, 1e-12))
    ln_m    = np.log(max(f_m,    1e-12))
    ln_next = np.log(max(f_next, 1e-12))

    delta = d_m - d_prev  # step size (negative when sweep is hi→lo)

    numerator = (
        (ln_m - ln_next) * (d_m ** 2 - d_prev ** 2)
        - (ln_m - ln_prev) * (d_m ** 2 - d_next ** 2)
    )
    denominator = 2.0 * delta * ((ln_m - ln_prev) + (ln_m - ln_next))

    if abs(denominator) < 1e-12:
        return float(d_m)

    # Use min/max so clip bounds are valid regardless of sweep direction.
    # np.clip(x, a, b) is undefined when a > b; the hi→lo sweep gives
    # d_prev > d_next, so without this guard the result is always clamped to d_next.
    d_lo = min(d_prev, d_next)
    d_hi = max(d_prev, d_next)
    return float(np.clip(numerator / denominator, d_lo, d_hi))


def fit_parabola(
    xs: npt.NDArray[np.float64],
    ys: npt.NDArray[np.float64],
) -> tuple[float, float, float]:
    """Fit a parabola ``y = a*(x - x0)^2 + y0`` by least squares.

    Args:
        xs: Independent variable values.
        ys: Dependent variable values.

    Returns:
        ``(x0, y0, a)`` — vertex coordinates and curvature.
    """
    A = np.column_stack([xs * xs, xs, np.ones_like(xs)])
    coeffs, *_ = np.linalg.lstsq(A, ys, rcond=None)
    a2, b, c = coeffs
    x0 = -b / (2 * a2)
    y0 = a2 * x0 * x0 + b * x0 + c
    return x0, y0, a2


def _bounding_box_from_corners(
    corners: npt.NDArray[np.float64],
    pad: int,
    frame_h: int,
    frame_w: int,
) -> tuple[int, int, int, int]:
    """Return an ``(x, y, w, h)`` ROI around tag corners, clamped to the frame."""
    x_min = int(np.floor(corners[:, 0].min())) - pad
    y_min = int(np.floor(corners[:, 1].min())) - pad
    x_max = int(np.ceil(corners[:, 0].max())) + pad
    y_max = int(np.ceil(corners[:, 1].max())) + pad
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(frame_w, x_max)
    y_max = min(frame_h, y_max)
    return x_min, y_min, x_max - x_min, y_max - y_min


def _find_peak(
    ds: list[float],
    ms: list[float],
    peak_idx: int,
    d_min: float,
    d_max: float,
) -> float:
    """Resolve the sub-step peak diopter from a discrete focus curve.

    Cascade: Gaussian log interpolation → parabola fit → argmax.
    """
    n = len(ds)

    # 1. Gaussian log interpolation (needs a neighbour on each side, all > 0)
    if 0 < peak_idx < n - 1:
        f_prev = ms[peak_idx - 1]
        f_m    = ms[peak_idx]
        f_next = ms[peak_idx + 1]
        if f_prev > 0 and f_m > 0 and f_next > 0:
            d_est = _gaussian_log_peak(
                ds[peak_idx - 1], ds[peak_idx], ds[peak_idx + 1],
                f_prev, f_m, f_next,
            )
            if d_min <= d_est <= d_max:
                return d_est

    # 2. Parabola fit over a window around the peak
    xs = np.array(ds, dtype=np.float64)
    ys = np.array(ms, dtype=np.float64)
    half = max(2, n // 6)
    left  = max(0, peak_idx - half)
    right = min(n, peak_idx + half + 1)
    if right - left >= 3:
        x0, _y0, a = fit_parabola(xs[left:right], ys[left:right])
        if a < 0 and d_min <= x0 <= d_max:
            return x0

    # 3. Argmax fallback
    return float(ds[peak_idx])


def _save_debug_outputs(
    tag_id: int,
    ds_c: list[float],
    ms_c: list[float],
    runs_hi2lo: list[tuple[list[float], list[float]]],
    peaks_hi2lo: list[float],
    runs_lo2hi: list[tuple[list[float], list[float]]],
    peaks_lo2hi: list[float],
    best_d_hi2lo: float,
    best_d_lo2hi: float,
    best_d: float,
    peak_m: float,
    roi_patch: npt.NDArray[np.uint8],
    debug_dir: Path,
    timestamp: str,
) -> None:
    """Save focus-curve plot (coarse + both fine directions, all repeats) and ROI crop."""
    stem = f"debug_tag{tag_id}_{timestamp}"
    hysteresis = abs(best_d_hi2lo - best_d_lo2hi)
    n_reps = len(runs_hi2lo)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ds_c, ms_c, "o-", color="steelblue", markersize=4, label="coarse")

    for i, ((ds, ms), peak) in enumerate(zip(runs_hi2lo, peaks_hi2lo)):
        alpha = 1.0 - 0.6 * i / max(n_reps - 1, 1)
        label = "fine ↓ (hi→lo)" if i == 0 else f"fine ↓ run {i + 1}"
        ax.plot(ds, ms, "s-", color="darkorange", alpha=alpha, markersize=4, label=label)
        if n_reps > 1:
            ax.axvline(peak, color="darkorange", alpha=alpha * 0.7, linestyle=":", linewidth=1.0)

    for i, ((ds, ms), peak) in enumerate(zip(runs_lo2hi, peaks_lo2hi)):
        alpha = 1.0 - 0.6 * i / max(n_reps - 1, 1)
        label = "fine ↑ (lo→hi)" if i == 0 else f"fine ↑ run {i + 1}"
        ax.plot(ds, ms, "^-", color="seagreen", alpha=alpha, markersize=4, label=label)
        if n_reps > 1:
            ax.axvline(peak, color="seagreen", alpha=alpha * 0.7, linestyle=":", linewidth=1.0)

    ax.axvline(best_d_hi2lo, color="darkorange", linestyle=":", linewidth=1.2,
               label=f"↓ mean = {best_d_hi2lo:.3f} D")
    ax.axvline(best_d_lo2hi, color="seagreen",   linestyle=":", linewidth=1.2,
               label=f"↑ mean = {best_d_lo2hi:.3f} D")
    ax.axvline(best_d, color="red", linestyle="--", linewidth=1.8,
               label=f"avg = {best_d:.3f} D  (Δ={hysteresis:.3f})")
    ax.set_xlim(best_d - 1.0, best_d + 1.0)
    ax.set_xlabel("Diopter (D)")
    ax.set_ylabel("Focus metric (Tenengrad)")
    ax.set_title(
        f"Tag {tag_id}  —  best {best_d:.3f} D  "
        f"(Δ hysteresis = {hysteresis:.3f} D{'  ⚠' if hysteresis > HYSTERESIS_THRESH else ''})"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = debug_dir / f"{stem}_curve.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"    [debug] curve  → {plot_path}")

    roi_path = debug_dir / f"{stem}_roi.jpg"
    cv2.imwrite(str(roi_path), roi_patch)
    print(f"    [debug] ROI    → {roi_path}")


def sweep_all_tags(
    lens: _Lens,
    focus_cam: _FocusCam,
    tag_family: str,
    diopter_range: tuple[float, float],
    n_coarse: int = 20,
    n_fine: int = 20,
    n_fine_repeats: int = 1,
    settle_s: float = 0.05,
    debug: bool = False,
    debug_dir: Path | None = None,
) -> dict[int, tuple[float, float, float, float, list[float], list[float]]]:
    """Sweep the liquid lens and find the best-focus diopter for each tag.

    **Coarse pass** — sweeps the full diopter range. At each step the XIMEA
    frame is grabbed and AprilTags are detected; per-tag bounding-box ROIs are
    tracked and focus metrics computed. Tags that are not detectable at a given
    diopter are still measured using their last known ROI.

    **Fine pass** — for each tag, sweeps a narrow window (±3 coarse steps)
    around its coarse peak in both directions (hi→lo then lo→hi) to capture
    hysteresis. Both per-direction peaks and their average are returned.

    Args:
        lens: Optotune lens instance.
        focus_cam: :class:`~liquid_lens_calibration.focus_camera.XimeaFocusCamera`.
        tag_family: Key into :data:`~liquid_lens_calibration.triangulate.TAG_FAMILIES`.
        diopter_range: ``(min_diopter, max_diopter)``.
        n_coarse: Steps in the coarse sweep.
        n_fine: Steps in the fine sweep per tag.
        settle_s: Seconds to wait after each diopter change.
        debug: If ``True``, save a focus-curve plot and ROI crop per tag.
        debug_dir: Directory for debug files. Defaults to ``./debug``.

    Returns:
        ``{tag_id: (best_diopter_avg, best_d_hi2lo, best_d_lo2hi,
        peak_metric, all_diopters, all_metrics)}``.
        Only tags detected in the XIMEA during the coarse sweep are returned.
        Returns an empty dict if no tags were found.
    """
    d_min, d_max = diopter_range
    dictionary_type = TAG_FAMILIES.get(tag_family, cv2.aruco.DICT_APRILTAG_36H11)

    out_dir = debug_dir if debug_dir is not None else Path("debug")
    ts = datetime.now().strftime("%H%M%S")
    if debug:
        out_dir.mkdir(parents=True, exist_ok=True)

    per_tag_roi: dict[int, tuple[int, int, int, int]] = {}
    per_tag_meas: dict[int, list[tuple[float, float]]] = defaultdict(list)

    # CLAHE equaliser — created once, applied to every frame before detection.
    # Normalises local contrast so the tag border remains detectable even when
    # the overall image is defocused or unevenly lit.
    _clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _detect_ximea(
        frame: npt.NDArray[np.uint8],
    ) -> tuple[list[npt.NDArray[np.float64]], npt.NDArray[np.int32] | None]:
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        enhanced = _clahe.apply(gray)
        return detect_apriltags(enhanced, dictionary_type, robust=True)

    # --- Coarse sweep ---
    # Frames are retained so metrics can be computed retroactively for the full
    # d_min→d_max range once every tag's ROI is known (tags are only detectable
    # near their focus diopter, so ROIs are not available for early steps).
    coarse_frames: list[tuple[float, npt.NDArray[np.uint8]]] = []
    _debug_save_every = max(1, n_coarse // 5)

    for _step_i, d in enumerate(np.linspace(d_min, d_max, n_coarse)):
        lens.set_diopter(float(d))
        time.sleep(settle_s)
        frame = focus_cam.grab_full_frame()
        frame_h, frame_w = frame.shape[:2]
        coarse_frames.append((float(d), frame))

        show_preview(frame, f"Coarse sweep  {d:+.2f} D  ({_step_i + 1}/{n_coarse})")
        cv2.waitKey(1)

        if debug and _step_i % _debug_save_every == 0:
            small = cv2.resize(frame, (frame_w // 2, frame_h // 2),
                               interpolation=cv2.INTER_AREA)
            frame_path = out_dir / f"ximea_{ts}_step{_step_i:02d}_{d:+.2f}D.jpg"
            cv2.imwrite(str(frame_path), small)

        corners_list, ids = _detect_ximea(frame)
        if ids is not None:
            for i, tag_id in enumerate(ids.flatten()):
                tid = int(tag_id)
                corners_4 = corners_list[i].reshape(4, 2)
                roi = _bounding_box_from_corners(
                    corners_4, pad=10, frame_h=frame_h, frame_w=frame_w
                )
                if roi[2] > 0 and roi[3] > 0:
                    per_tag_roi[tid] = roi

    if not per_tag_roi:
        return {}

    # Retroactively compute the full coarse metric curve for each tag using its
    # final detected ROI. This covers all n_coarse steps from d_min to d_max
    # even for the steps where the tag was too blurry to detect.
    for tag_id, (x, y, bw, bh) in per_tag_roi.items():
        for d, frame in coarse_frames:
            patch = frame[y : y + bh, x : x + bw]
            per_tag_meas[tag_id].append((float(d), focus_metric(patch)))
    coarse_frames.clear()  # release frame memory before fine sweeps

    coarse_step = (d_max - d_min) / max(n_coarse - 1, 1)
    results: dict[int, tuple[float, float, float, list[float], list[float]]] = {}

    # --- Per-tag fine sweep ---
    for tag_id, meas in per_tag_meas.items():
        roi = per_tag_roi.get(tag_id)
        if roi is None:
            continue

        ds_c = [m[0] for m in meas]
        ms_c = [m[1] for m in meas]
        coarse_peak_d = ds_c[int(np.argmax(ms_c))]

        fine_half = 3.0 * coarse_step
        fine_min = max(d_min, coarse_peak_d - fine_half)
        fine_max = min(d_max, coarse_peak_d + fine_half)

        # Pre-settle at fine_max before the hi→lo sweeps.
        lens.set_diopter(float(fine_max))
        time.sleep(max(settle_s * 10, 1.0))

        # Fine sweep: hi→lo, repeated n_fine_repeats times.
        runs_hi2lo: list[tuple[list[float], list[float]]] = []
        peak_frame: npt.NDArray[np.uint8] | None = None
        best_m_so_far = -1.0
        for rep in range(n_fine_repeats):
            if rep > 0:
                lens.set_diopter(float(fine_max))
                time.sleep(max(settle_s * 10, 1.0))
            ds_run: list[float] = []
            ms_run: list[float] = []
            for _step_j, d in enumerate(np.linspace(fine_max, fine_min, n_fine)):
                lens.set_diopter(float(d))
                time.sleep(settle_s)
                patch = focus_cam.grab_roi_frame(roi)
                ds_run.append(float(d))
                m = focus_metric(patch)
                ms_run.append(m)
                if m >= best_m_so_far:
                    best_m_so_far = m
                    peak_frame = patch
                show_preview(patch,
                             f"Fine ↓  tag {tag_id}  rep {rep + 1}/{n_fine_repeats}  "
                             f"{d:+.2f} D  ({_step_j + 1}/{n_fine})  metric={m:.0f}")
                cv2.waitKey(1)
            runs_hi2lo.append((ds_run, ms_run))

        # Pre-settle at fine_min before the lo→hi sweeps.
        lens.set_diopter(float(fine_min))
        time.sleep(max(settle_s * 10, 1.0))

        # Fine sweep: lo→hi, repeated n_fine_repeats times.
        runs_lo2hi: list[tuple[list[float], list[float]]] = []
        for rep in range(n_fine_repeats):
            if rep > 0:
                lens.set_diopter(float(fine_min))
                time.sleep(max(settle_s * 10, 1.0))
            ds_run = []
            ms_run = []
            for _step_j, d in enumerate(np.linspace(fine_min, fine_max, n_fine)):
                lens.set_diopter(float(d))
                time.sleep(settle_s)
                patch = focus_cam.grab_roi_frame(roi)
                ds_run.append(float(d))
                m = focus_metric(patch)
                ms_run.append(m)
                if m >= best_m_so_far:
                    best_m_so_far = m
                    peak_frame = patch
                show_preview(patch,
                             f"Fine ↑  tag {tag_id}  rep {rep + 1}/{n_fine_repeats}  "
                             f"{d:+.2f} D  ({_step_j + 1}/{n_fine})  metric={m:.0f}")
                cv2.waitKey(1)
            runs_lo2hi.append((ds_run, ms_run))

        # Find peak for each run, then average across repeats.
        peaks_hi2lo = [
            _find_peak(ds, ms, int(np.argmax(ms)), d_min, d_max)
            for ds, ms in runs_hi2lo
        ]
        peaks_lo2hi = [
            _find_peak(ds, ms, int(np.argmax(ms)), d_min, d_max)
            for ds, ms in runs_lo2hi
        ]
        best_d_hi2lo = float(np.mean(peaks_hi2lo))
        best_d_lo2hi = float(np.mean(peaks_lo2hi))
        hysteresis = abs(best_d_hi2lo - best_d_lo2hi)
        best_d = (best_d_hi2lo + best_d_lo2hi) / 2.0
        peak_m = float(max(max(ms) for _, ms in runs_hi2lo + runs_lo2hi))

        if n_fine_repeats > 1:
            std_hi = float(np.std(peaks_hi2lo))
            std_lo = float(np.std(peaks_lo2hi))
            warn = "  [warn]" if hysteresis > HYSTERESIS_THRESH else ""
            print(f"   {warn} tag {tag_id}: ↓{best_d_hi2lo:.3f}±{std_hi:.3f}  "
                  f"↑{best_d_lo2hi:.3f}±{std_lo:.3f} D  (Δ{hysteresis:.3f}) → {best_d:.3f} D")
        elif hysteresis > HYSTERESIS_THRESH:
            print(f"    [warn] tag {tag_id}: hysteresis {hysteresis:.3f} D "
                  f"(↓{best_d_hi2lo:.3f}  ↑{best_d_lo2hi:.3f}) — averaging → {best_d:.3f} D")
        else:
            print(f"    tag {tag_id}: fine ↓{best_d_hi2lo:.3f}  ↑{best_d_lo2hi:.3f} D "
                  f"(Δ{hysteresis:.3f}) → {best_d:.3f} D")

        all_ds = ds_c + [d for ds, _ in runs_hi2lo + runs_lo2hi for d in ds]
        all_ms = ms_c + [m for _, ms in runs_hi2lo + runs_lo2hi for m in ms]
        results[tag_id] = (best_d, best_d_hi2lo, best_d_lo2hi, peak_m, all_ds, all_ms)

        if debug and peak_frame is not None:
            _save_debug_outputs(
                tag_id, ds_c, ms_c,
                runs_hi2lo, peaks_hi2lo,
                runs_lo2hi, peaks_lo2hi,
                best_d_hi2lo, best_d_lo2hi,
                best_d, peak_m, peak_frame, out_dir, ts,
            )

    return results
