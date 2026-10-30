# Copyright (C) 2025, SynDefect3D.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""A camera sensor between the render and the compositor look.

Cycles hands the compositor scene-referred radiance and AgX maps it to display
with a shoulder that never quite reaches white. Nothing in that path is a
camera, so two real behaviours are missing and both of them keep defects
visible when they should not be:

* **Highlights compress instead of clipping.** A wall lit forty times over
  middle grey still lands at code 237-246 under AgX, and a crack inside it
  survives as a three-to-five code difference. A real sensor fills its wells,
  every pixel in the blown region reads the same value, and the crack is gone.
* **There is no noise.** Cycles plus OIDN is noiseless at any light level, so a
  two percent contrast crack is as readable in a 5 lux room as in daylight. A
  real phone raises ISO in the dark and buries that contrast in shot noise.

So this module puts a sensor in front of the look. Scene radiance is metered,
converted to photoelectrons, given shot and read noise, clipped at the full
well, and amplified by whatever gain metering could not get from the shutter:

    e        = L * g_shutter * full_well
    e_noisy  = e + N(0, sqrt(e + read^2))
    out      = min(e_noisy, full_well) / full_well * g_iso, clipped to white

Gain is split because it matters where it is applied. A longer exposure
collects more photons and is therefore quieter; ISO only amplifies what was
already collected, noise included. A dark room exceeds the handheld shutter
limit, the rest of the gain becomes ISO, and the image comes out grey and
noisy exactly as a phone photo of a dark room does.

The saturation point only maps to display white under a view transform that
takes linear 1.0 to 1.0 -- ``Standard``. Under AgX the sensor still clips, but
the clipped patch lands near code 247 instead of 255. It is flat either way,
which is what hides the defect; ``sensor_realism.gin`` switches the transform
so it is also white.

``sensor_transfer`` below is the same arithmetic in numpy. The node graph
mirrors it, the tests check it, and ``tools/measure_camera_response.py``
uses it to predict what a render should produce.
"""

import logging

import bpy
import gin
import numpy as np

from cwg_infinigen.core.nodes.node_info import Nodes
from cwg_infinigen.core.nodes.node_wrangler import NodeWrangler
from cwg_infinigen.core.util.math import int_hash

logger = logging.getLogger(__name__)

# Electrons at saturation for a well-lit phone pixel, and the read floor that
# sets how dark a scene can get before the image is noise. 12k/3e- puts mid
# grey at roughly 2% noise in good light, which is where phone JPEGs sit.
DEFAULT_FULL_WELL_E = 12000.0
DEFAULT_READ_NOISE_E = 3.0


def sensor_transfer(
    linear: np.ndarray,
    gain_shutter: float = 1.0,
    gain_iso: float = 1.0,
    full_well_e: float = DEFAULT_FULL_WELL_E,
    read_noise_e: float = DEFAULT_READ_NOISE_E,
    white_level: float = 1.0,
    noise: np.ndarray | None = None,
) -> np.ndarray:
    """Scene-linear radiance -> normalised sensor signal.

    ``noise`` is standard-normal samples shaped like ``linear``; pass None for
    the noiseless transfer (the clipping behaviour on its own).
    """
    photons = np.maximum(np.asarray(linear, dtype=np.float64), 0.0)
    photons = photons * float(gain_shutter) * float(full_well_e)
    if noise is not None:
        sigma = np.sqrt(photons + float(read_noise_e) ** 2)
        photons = photons + sigma * np.asarray(noise, dtype=np.float64)
    # The well fills before any amplification: this is the clip that flattens
    # blown highlights, and no amount of ISO recovers what it discarded.
    photons = np.minimum(photons, float(full_well_e))
    signal = photons / float(full_well_e) * float(gain_iso)
    return np.clip(signal, 0.0, float(white_level))


def split_gain(total_gain: float, shutter_gain_limit: float) -> tuple[float, float]:
    """Split metering gain into photon-collecting and amplifying parts."""
    total = max(float(total_gain), 1e-6)
    shutter = min(total, max(float(shutter_gain_limit), 1e-6))
    return shutter, total / shutter


def _noise_image(width: int, height: int, seed: int) -> bpy.types.Image:
    """A float image of standard-normal samples, one draw per pixel channel.

    Generated rather than taken from the legacy NOISE texture so the
    distribution is actually Gaussian, the seed is recorded, and the same
    values can be reproduced in numpy when checking a render.
    """
    name = f"SynDefectSensorNoise_{seed:08x}"
    existing = bpy.data.images.get(name)
    if existing is not None:
        if tuple(existing.size) == (width, height):
            return existing
        bpy.data.images.remove(existing)

    img = bpy.data.images.new(name, width=width, height=height, float_buffer=True)
    # Raw passthrough: these are signed noise samples, not a picture, and an
    # sRGB decode on load would fold the negative half away.
    img.colorspace_settings.name = "Non-Color"
    rng = np.random.default_rng(seed)
    pixels = np.empty((height * width, 4), dtype=np.float32)
    pixels[:, :3] = rng.standard_normal((height * width, 3), dtype=np.float32)
    pixels[:, 3] = 1.0
    img.pixels.foreach_set(pixels.reshape(-1))
    return img


def _render_pixel_size() -> tuple[int, int]:
    render = bpy.context.scene.render
    scale = render.resolution_percentage / 100.0
    return max(1, int(render.resolution_x * scale)), max(1, int(render.resolution_y * scale))


def _math(nw: NodeWrangler, operation: str, a, b=None):
    inputs = {0: a} if b is None else {0: a, 1: b}
    return nw.new_node(
        Nodes.CompositorMath, input_kwargs=inputs, attrs={"operation": operation}
    ).outputs[0]


def _metered_gain(nw: NodeWrangler, source, target_grey, min_gain, max_gain, ev):
    """Auto-exposure: whole-frame luminance mean driven to ``target_grey``.

    Whole-frame rather than centre-weighted. It is the crude version of what a
    phone meter does, and it is the part that matters here: it decides how much
    gain a dark room needs, and therefore how noisy the photo of it is.
    """
    levels = nw.new_node(
        Nodes.CompositorLevels,
        input_kwargs={"Image": source},
        attrs={"channel": "LUMINANCE"},
    )
    mean = _math(nw, "MAXIMUM", levels.outputs["Mean"], 1e-5)
    gain = _math(nw, "DIVIDE", target_grey, mean)
    gain = _math(nw, "MINIMUM", gain, max_gain)
    gain = _math(nw, "MAXIMUM", gain, min_gain)
    if abs(float(ev)) > 1e-6:
        gain = _math(nw, "MULTIPLY", gain, float(2.0 ** float(ev)))
    return gain


@gin.configurable
def apply_auto_white_balance(
    nw: NodeWrangler,
    source,
    enabled: bool = True,
    max_correction: float = 3.0,
    preserve_exposure: bool = True,
    strength: float = 1.0,
):
    """Grey-world auto white balance, in linear light, before the sensor.

    `strength` (0..1) raises each channel gain to that power: 1.0 is full grey
    world (default, unchanged graph); lower values only partly neutralise, as
    phone AWB does on a genuinely coloured wall instead of draining it to grey.

    Why this exists: the wall palette in this repo is already neutral - seven
    paints from bright white to mid grey, every one at R/B 1.01-1.07, with
    `colored_prob = 0.0` so the tinted branch never fires. Indoor lamps are
    sampled at clip_gaussian(4700, 700, 3500, 6500) K, which is warm against the
    sRGB D65 white point, and nothing downstream corrected for it. Measured on a
    rendered ceiling, a neutral wall came out at R/B = 1.93, about +80 code
    values of orange. Every scene got the same cast, so re-rendering more scenes
    could never diversify it away.

    A real phone does not have this problem because it white balances: that is
    why a photo of a tungsten-lit room still shows a white wall. This is the
    chromatic counterpart of the auto-exposure already in apply_sensor_response,
    and deliberately the same crude class of algorithm - grey world - because
    the point is to remove a systematic illuminant cast, not to model a
    particular vendor's AWB.

    Grey world neutralises the frame's *average*, so it will also drain a scene
    genuinely dominated by one colour. `max_correction` bounds how far it can
    push any channel; with neutral plaster and timber floors the correction
    needed here is well inside it.

    `preserve_exposure` divides out the luminance-weighted mean gain, so white
    balance changes chromaticity only and leaves metering, clipping and the
    measured contrast numbers alone.
    """
    if not enabled:
        return source

    means = []
    for channel in ("RED", "GREEN", "BLUE"):
        levels = nw.new_node(
            Nodes.CompositorLevels,
            input_kwargs={"Image": source},
            attrs={"channel": channel},
        )
        means.append(_math(nw, "MAXIMUM", levels.outputs["Mean"], 1e-5))

    grey = _math(nw, "ADD", _math(nw, "ADD", means[0], means[1]), means[2])
    grey = _math(nw, "DIVIDE", grey, 3.0)

    lo = 1.0 / float(max_correction)
    gains = []
    for mean in means:
        gain = _math(nw, "DIVIDE", grey, mean)
        gain = _math(nw, "MINIMUM", gain, float(max_correction))
        gain = _math(nw, "MAXIMUM", gain, lo)
        if strength != 1.0:
            gain = _math(nw, "POWER", gain, float(strength))
        gains.append(gain)

    if preserve_exposure:
        lum = _math(nw, "MULTIPLY", gains[0], 0.2126)
        lum = _math(nw, "ADD", lum, _math(nw, "MULTIPLY", gains[1], 0.7152))
        lum = _math(nw, "ADD", lum, _math(nw, "MULTIPLY", gains[2], 0.0722))
        lum = _math(nw, "MAXIMUM", lum, 1e-5)
        gains = [_math(nw, "DIVIDE", g, lum) for g in gains]

    channels = nw.new_node(Nodes.CompSeparateColor, input_kwargs={"Image": source})
    balanced = [
        _math(nw, "MULTIPLY", channels.outputs[i], gains[i]) for i in range(3)
    ]
    return nw.new_node(
        Nodes.CompCombineColor,
        input_kwargs={0: balanced[0], 1: balanced[1], 2: balanced[2], 3: 1.0},
    ).outputs["Image"]


@gin.configurable
def apply_sensor_response(
    nw: NodeWrangler,
    source,
    enabled: bool = False,
    auto_exposure: bool = True,
    target_grey: float = 0.18,
    ev_compensation: float = 0.0,
    # A phone can stop down a long way, so metering is allowed to darken far
    # more than it brightens. Clamping the bottom too early would blow out a
    # merely bright wall, and real overexposure comes from scene dynamic range
    # -- a window against a room -- not from overall level.
    min_exposure_gain: float = 1.0 / 4096.0,
    max_exposure_gain: float = 256.0,
    shutter_gain_limit: float = 8.0,
    full_well_e: float = DEFAULT_FULL_WELL_E,
    read_noise_e: float = DEFAULT_READ_NOISE_E,
    highlight_headroom_stops: float = 2.47,
    noise: bool = True,
    noise_seed: int | None = None,
):
    """Insert the sensor between ``source`` and the rest of the compositor.

    Returns the new socket, or ``source`` untouched when disabled, so callers
    can drop it in front of an existing chain.
    """
    if not enabled:
        return source

    # Headroom is expressed against middle grey because that is how exposure is
    # reasoned about: 2.47 stops is 0.18 -> 1.0, a phone's roughly three stops
    # of highlight latitude before a window goes white.
    white_level = float(target_grey) * float(2.0 ** float(highlight_headroom_stops))

    if auto_exposure:
        gain = _metered_gain(
            nw, source, target_grey, min_exposure_gain, max_exposure_gain, ev_compensation
        )
        # The shutter cannot run past handheld, so gain above the limit is ISO:
        # amplification of a photon count that was never collected.
        gain_shutter = _math(nw, "MINIMUM", gain, float(shutter_gain_limit))
        gain_iso = _math(nw, "DIVIDE", gain, gain_shutter)
    else:
        shutter, iso = split_gain(2.0 ** float(ev_compensation), shutter_gain_limit)
        gain_shutter, gain_iso = float(shutter), float(iso)

    if noise:
        width, height = _render_pixel_size()
        if noise_seed is None:
            scene = bpy.context.scene
            cam = scene.camera
            # Pass index included so each domain-randomisation pass of a rig is a
            # different photograph rather than the same one re-tonemapped: two
            # exposures of one scene do not share a noise realisation.
            noise_seed = int_hash(
                (
                    int(scene.frame_current),
                    cam.name if cam else "nocam",
                    int(scene.get("syndefect_resample_idx", 0)),
                )
            ) % (2**31)
        noise_node = nw.new_node(
            Nodes.CompositorImage,
            attrs={"image": _noise_image(width, height, int(noise_seed))},
        )
        noise_channels = nw.new_node(
            Nodes.CompSeparateColor, input_kwargs={"Image": noise_node.outputs["Image"]}
        )
        logger.info(
            "sensor response: %dx%d noise, seed %d, full well %.0f e-, read %.1f e-",
            width, height, noise_seed, full_well_e, read_noise_e,
        )
    else:
        noise_channels = None

    channels = nw.new_node(Nodes.CompSeparateColor, input_kwargs={"Image": source})
    read_variance = float(read_noise_e) ** 2
    out_channels = []
    for idx in range(3):
        lin = channels.outputs[idx]
        photons = _math(nw, "MULTIPLY", lin, gain_shutter)
        photons = _math(nw, "MULTIPLY", photons, float(full_well_e))
        photons = _math(nw, "MAXIMUM", photons, 0.0)
        if noise_channels is not None:
            variance = _math(nw, "ADD", photons, read_variance)
            sigma = _math(nw, "SQRT", variance)
            photons = _math(
                nw,
                "ADD",
                photons,
                _math(nw, "MULTIPLY", sigma, noise_channels.outputs[idx]),
            )
        photons = _math(nw, "MINIMUM", photons, float(full_well_e))
        signal = _math(nw, "DIVIDE", photons, float(full_well_e))
        signal = _math(nw, "MULTIPLY", signal, gain_iso)
        signal = _math(nw, "MINIMUM", signal, white_level)
        out_channels.append(_math(nw, "MAXIMUM", signal, 0.0))

    combined = nw.new_node(
        Nodes.CompCombineColor,
        input_kwargs={0: out_channels[0], 1: out_channels[1], 2: out_channels[2], 3: 1.0},
    )
    # Rescale so saturation is display white. Under Standard this is exactly a
    # clipped sensor; under AgX the shoulder still softens the top.
    if abs(white_level - 1.0) > 1e-6:
        return nw.new_node(
            Nodes.Exposure,
            input_kwargs={
                "Image": combined.outputs["Image"],
                "Exposure": float(-np.log2(white_level)),
            },
        ).outputs[0]
    return combined.outputs["Image"]
