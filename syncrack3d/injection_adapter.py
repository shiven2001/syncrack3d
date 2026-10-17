"""Adapter from explicit controlled placements to Infinigen defect injection.

This module does not create defects.  It is the future integration boundary:
the caller supplies already-created defect card objects and the exact host for
each one.  The adapter builds the small relation state that ``run_injection``
expects, avoiding nearest-wall guesses while reusing the existing injection,
support validation, substrate checks, displacement, and AOV machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import bpy


@dataclass(frozen=True)
class DefectBinding:
    defect: bpy.types.Object
    host: bpy.types.Object
    surface: str = "wall"


@dataclass
class _Relation:
    target_name: str


@dataclass
class _ObjectState:
    obj: bpy.types.Object
    relations: list[_Relation] = field(default_factory=list)
    tags: set = field(default_factory=set)


@dataclass
class _ControlledState:
    objs: dict[str, _ObjectState] = field(default_factory=dict)


def _state_for_bindings(bindings):
    from cwg_infinigen.core import tags as t

    state = _ControlledState()
    host_by_name = {}
    for index, binding in enumerate(bindings):
        host = binding.host
        if not host.name.endswith((".wall", ".ceiling", ".floor")):
            raise ValueError(
                f"Controlled injection host must end in a surface suffix: {host.name}"
            )
        room_name = host.name.rsplit(".", 1)[0]
        host_by_name[host.name] = host
        state.objs.setdefault(
            room_name,
            _ObjectState(obj=host, tags={t.Semantics.Room}),
        )
        binding.defect["syndefect_surface"] = binding.surface
        state.objs[f"defect_{index:04d}"] = _ObjectState(
            obj=binding.defect,
            relations=[_Relation(target_name=room_name)],
        )
    return state, host_by_name


def run_controlled_injection(
    bindings: list[DefectBinding],
    *,
    manifest_path: str | Path | None = None,
    scene_id: str | None = None,
    scene_seed: int | None = None,
    dicing_rate: float = 2.0,
    max_per_host: int = 6,
) -> int:
    """Inject explicit bindings using the existing Infinigen implementation.

    Nothing calls this during base-scene generation.  It becomes active only
    when the caller supplies defect objects.
    """
    if not bindings:
        return 0
    from cwg_infinigen.assets.defect_injection import run_injection, write_defect_manifest

    state, host_by_name = _state_for_bindings(bindings)
    count = run_injection(
        state=state,
        host_by_name=host_by_name,
        enabled=True,
        hide_planes=True,
        dicing_rate=dicing_rate,
        max_per_host=max_per_host,
        drop_over_cap=True,
        injection_only=True,
        enforce_substrates=True,
        validate_support=True,
        allow_nearest_host_fallback=False,
        require_qa_view=False,
        retarget_dropped_cameras=False,
        fail_scene_on_injection_error=True,
    )
    if manifest_path is not None:
        write_defect_manifest(
            manifest_path,
            scene_id=scene_id,
            scene_seed=scene_seed,
        )
    return count
