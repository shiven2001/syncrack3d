"""SynCrack3D: controlled walls with procedural cracks.

The package deliberately keeps Blender imports out of its top level so scene
specifications can be sampled and tested with ordinary Python.  Geometry,
materials, and the optional Infinigen injection adapter are loaded only by the
Blender entry point.
"""

from .spec import GENERATOR_VERSION, MaterialSpec, SceneSpec, sample_scene_spec

__all__ = [
    "GENERATOR_VERSION",
    "MaterialSpec",
    "SceneSpec",
    "sample_scene_spec",
]
