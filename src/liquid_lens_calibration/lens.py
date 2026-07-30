"""Thin wrapper over the optotune-lens package, with driver auto-detection."""

from typing import Union

from optotune_lens import ICC1C, Lens, LensError
from optotune_lens.lens import OperatingMode

LensDriver = Union[Lens, ICC1C]


def _detect_driver(port: str) -> LensDriver:
    """Identify and connect to whichever Optotune controller is on `port`.

    The Lens Driver 4 and ICC-1C speak incompatible handshake protocols, so
    identifying which one is connected just means requesting each one's
    handshake in turn — both `Lens.__init__` and `ICC1C.__init__` already
    perform their own handshake and raise `LensError` (via a serial read
    timeout) if the device on the other end doesn't answer as expected.

    Args:
        port: Serial port device path.

    Returns:
        A connected `Lens` or `ICC1C` instance.

    Raises:
        LensError: If neither driver responds on `port`.
    """
    errors: list[str] = []
    for driver_cls in (Lens, ICC1C):
        try:
            driver = driver_cls(port)
        except LensError as e:
            errors.append(f"{driver_cls.__name__}: {e}")
            continue
        print(f"Detected {driver_cls.__name__} controller on {port}")
        return driver

    raise LensError(
        f"No supported Optotune controller responded on {port} "
        f"(tried Lens Driver 4 and ICC-1C). Check the cable and port, and "
        f"that the controller is powered on.\n" + "\n".join(f"  - {e}" for e in errors)
    )


def open_lens(port: str = "/dev/optotune_ld") -> tuple[LensDriver, tuple[float, float]]:
    """Auto-detect and open the Optotune controller on `port`, in focal-power mode.

    Args:
        port: Serial port device path.

    Returns:
        ``(lens_instance, (min_diopter, max_diopter))``.

    Raises:
        LensError: If no controller responds on `port`, or the mode switch
            or diopter-range query fails.
    """
    lens = _detect_driver(port)

    if isinstance(lens, ICC1C):
        # Simple Mode has no mode-switch; this just validates EEPROM presence.
        lens.to_focal_power_mode()
        d_min = lens.get_diopter_min()
        d_max = lens.get_diopter_max()
        if d_min is None or d_max is None:
            raise LensError(
                "ICC-1C did not report a focal power range for the connected lens"
            )
        return lens, (d_min, d_max)

    # Lens Driver 4. If already in focal-power mode (e.g. not power-cycled
    # between runs), skip the mode-switch command to avoid the spurious
    # error 72 ("already in this mode") returned by the firmware.
    if lens.mode == OperatingMode.FOCAL_POWER:
        assert lens.min_diopter is not None and lens.max_diopter is not None
        diopter_range = (lens.min_diopter, lens.max_diopter)
    else:
        diopter_range = lens.to_focal_power_mode()
        if lens.mode != OperatingMode.FOCAL_POWER:
            raise LensError(
                f"Failed to switch lens to focal power mode; current mode: {lens.mode}"
            )

    return lens, diopter_range
