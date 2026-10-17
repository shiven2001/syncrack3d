"""Pure camera-framing calculations, independent of Blender."""

import random


def sample_exposure_evs(seed, count, ev_range):
    """Independent stream: exposure changes cannot resample scene geometry."""
    if ev_range is None:
        return None
    low, high = ev_range
    if not -3 <= low <= high <= 3:
        raise ValueError("Exposure range must satisfy -3 <= low <= high <= 3")
    rng = random.Random(int(seed) ^ 0x455850)
    return [rng.uniform(low, high) for _ in range(count)]


def solve_closeup(extent_m, fill, focal_mm, sensor_mm, distance_range, focal_choices):
    """Use a longer lens when minimum focus distance would shrink the defect.

    Returns focal length, clamped distance, and nominal fronto-parallel fill.
    Fill describes the supplied extent, not segmented area or projected width
    after yaw/pitch. It must not be interpreted as a mask coverage guarantee.
    """
    low, high = distance_range
    if not (
        extent_m > 0
        and 0 < fill <= 2
        and focal_mm > 0
        and sensor_mm > 0
        and 0 < low <= high
    ):
        raise ValueError("Invalid framing geometry")
    choices = sorted(set([focal_mm] + [f for f in focal_choices if f >= focal_mm]))
    chosen = choices[-1]
    for candidate in choices:
        chosen = candidate
        if extent_m * candidate / (sensor_mm * fill) >= low:
            break
    distance = min(high, max(low, extent_m * chosen / (sensor_mm * fill)))
    achieved = extent_m * chosen / (sensor_mm * distance)
    return chosen, distance, achieved
