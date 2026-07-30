"""Thin wrapper over the optotune-lens package, with driver auto-detection."""

from typing import Optional, Union

from optotune_lens import ICC1C, Lens, LensError
from optotune_lens.lens import OperatingMode

LensDriver = Union[Lens, ICC1C]

# Fixed udev symlinks on this rig, one per controller model (see
# /etc/udev/rules.d/99-optotune*.rules). Both are probed when no --port is
# given, since either — or neither, or both plugged in at once — may be true.
_DEFAULT_PORTS = ("/dev/optotune_icc1c", "/dev/optotune_ld")


def _diopter_range(lens: LensDriver) -> tuple[float, float]:
    """Put `lens` in focal-power mode and return its (min, max) diopter range.

    Raises:
        LensError: If the mode switch or range query fails, or the reported
            range is degenerate (e.g. a Lens Driver 4 box with no lens head
            attached still answers the handshake, but reports 0 to 0 D).
    """
    if isinstance(lens, ICC1C):
        # Simple Mode has no mode-switch; this just validates EEPROM presence.
        lens.to_focal_power_mode()
        d_min = lens.get_diopter_min()
        d_max = lens.get_diopter_max()
        if d_min is None or d_max is None:
            raise LensError(
                "ICC-1C did not report a focal power range for the connected lens"
            )
    else:
        # Lens Driver 4. If already in focal-power mode (e.g. not
        # power-cycled between runs), skip the mode-switch command to avoid
        # the spurious error 72 ("already in this mode") returned by the
        # firmware.
        if lens.mode == OperatingMode.FOCAL_POWER:
            assert lens.min_diopter is not None and lens.max_diopter is not None
            d_min, d_max = lens.min_diopter, lens.max_diopter
        else:
            d_min, d_max = lens.to_focal_power_mode()
            if lens.mode != OperatingMode.FOCAL_POWER:
                raise LensError(
                    f"Failed to switch lens to focal power mode; current mode: {lens.mode}"
                )

    if d_max <= d_min:
        raise LensError(
            f"{type(lens).__name__} reported a degenerate diopter range "
            f"({d_min} to {d_max} D) — is a lens head attached to it?"
        )
    return d_min, d_max


def _try_connect(port: str, errors: list[str]) -> Optional[tuple[LensDriver, tuple[float, float]]]:
    """Try each supported controller class on `port`, appending failures to `errors`.

    The Lens Driver 4 and ICC-1C speak incompatible handshake protocols, so
    identifying which one is connected just means requesting each one's
    handshake in turn — both `Lens.__init__` and `ICC1C.__init__` already
    perform their own handshake and raise `LensError` if the device on the
    other end doesn't answer as expected, so identification just means
    catching that and trying the next class.

    Returns:
        ``(lens_instance, (min_diopter, max_diopter))`` on success, or
        ``None`` if nothing usable answered on `port`.
    """
    for driver_cls in (Lens, ICC1C):
        try:
            driver = driver_cls(port)
        except LensError as e:
            errors.append(f"{driver_cls.__name__} on {port}: {e}")
            continue

        try:
            diopter_range = _diopter_range(driver)
        except LensError as e:
            errors.append(f"{driver_cls.__name__} on {port}: {e}")
            driver.close()
            continue

        return driver, diopter_range

    return None


def open_lens(port: Optional[str] = None) -> tuple[LensDriver, tuple[float, float]]:
    """Auto-detect and open the connected Optotune controller, in focal-power mode.

    Args:
        port: Serial port device path. If omitted, probes each of
            `_DEFAULT_PORTS` in turn and uses whichever one has a controller
            with a lens attached.

    Returns:
        ``(lens_instance, (min_diopter, max_diopter))``.

    Raises:
        LensError: If no controller with a usable lens attached is found.
    """
    ports = (port,) if port is not None else _DEFAULT_PORTS

    errors: list[str] = []
    for candidate in ports:
        found = _try_connect(candidate, errors)
        if found is not None:
            lens, diopter_range = found
            print(f"Detected {type(lens).__name__} controller on {candidate}")
            return lens, diopter_range

    raise LensError(
        f"No Optotune lens controller found (tried: {', '.join(ports)}). "
        f"Check the cable and power, and that a lens head is attached.\n"
        + "\n".join(f"  - {e}" for e in errors)
    )
