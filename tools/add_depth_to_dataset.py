#!/usr/bin/env python3
"""Add Depth, WallRelief and Camera files to walls rendered before depth export existed.

RGB and labels are not touched. Each wall is rebuilt from its seed with
`build_dataset.py --depth-only` (1 sample, depth, position and index passes only) into
<chunk>/depth_rebuild, then checked against the released wall: every field of the rebuilt
scene.json except timing must equal the released one, and the rebuilt views must be the
released views. Walls that pass have their files moved into <chunk>/walls/frames/seed_<seed>/
and hard-linked into <chunk>/pack/{depth,wall_relief,cameras}/ under the pack's image names.
Walls that fail are listed and left without depth.

    python tools/add_depth_to_dataset.py --dataset /mnt/nvme_storage/syncrack3d_v1.30.0_dataset \
        --chunks test val train_a train_b --workers 4 --build-flags "--recipe unified"

Resumable: walls that already have Depth/ are skipped. Renders on the CPU (CUDA_VISIBLE_DEVICES="").
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FOLDERS = (("Depth", "depth", ".png"), ("WallRelief", "wall_relief", ".png"), ("Camera", "cameras", ".json"))
# Fields that legitimately differ between a full render and a depth-only rebuild.
NOT_COMPARED = {"timing_s", "depth_export", "depth_only", "camera_isp", "view_lighting"}
PACK_ID = re.compile(r"seed_(\d+)_rig(\d+)_resample(\d+)_frame(\d+)_camera_(\d+)$")


def seed_dir(root: Path, seed: int) -> Path:
    return root / f"seed_{seed:05d}"


def rebuild(blender_py, flags, seeds, out, log, tries=20):
    """build_dataset --depth-only for `seeds`; Blender can segfault mid-run, --resume carries on."""
    cmd = [blender_py, str(REPO / "syncrack3d/build_dataset.py"), *shlex.split(flags), "--depth-only",
           "--seeds", ",".join(map(str, seeds)), "--output", str(out), "--keep-going", "--resume"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    for _ in range(tries):
        with open(log, "a") as fh:
            if subprocess.run(cmd, cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode == 0:
                return
    raise RuntimeError(f"depth-only rebuild kept failing, see {log}")


def check(released: Path, rebuilt: Path, frames: Path, new_frames: Path) -> str | None:
    """None if the rebuilt wall is the released wall, else the reason it is not."""
    if not (rebuilt / "scene.json").is_file():
        return "not rebuilt"
    a, b = (json.loads((p / "scene.json").read_text()) for p in (released, rebuilt))
    differ = sorted(k for k in (set(a) | set(b)) - NOT_COMPARED if a.get(k) != b.get(k))
    if differ:
        return f"scene.json differs in {differ}"
    views = {p.stem[len("Image_"):] for p in (frames / "Image/camera_0").glob("Image_*.png")}
    for folder, _, ext in FOLDERS:
        got = {p.stem[len(folder) + 1:] for p in (new_frames / folder / "camera_0").glob(f"*{ext}")}
        if got != views:
            return f"{folder} views {sorted(got)} != released {sorted(views)}"
    return None


def link_pack(chunk: Path) -> int:
    """Hard-link each pack image's depth files under the image's name; returns links made."""
    pack, made = chunk / "pack", 0
    for img in sorted((pack / "images").glob("*.png")):
        m = PACK_ID.match(img.stem)
        if not m:
            continue
        seed, rig, resample, frame, cam = m.groups()
        stem = f"{rig}_{resample}_{frame}_{cam}"
        frames = seed_dir(chunk / "walls/frames", int(seed))
        for folder, out_name, ext in FOLDERS:
            src = frames / folder / f"camera_{cam}" / f"{folder}_{stem}{ext}"
            dst = pack / out_name / f"{img.stem}{ext}"
            if src.is_file() and not dst.exists():
                dst.parent.mkdir(exist_ok=True)
                os.link(src, dst)
                made += 1
    return made


def add_chunk(chunk: Path, args) -> dict:
    walls, frames_root, scratch = chunk / "walls", chunk / "walls/frames", chunk / "depth_rebuild"
    seeds = sorted(int(p.name[5:]) for p in walls.glob("seed_*") if (p / "scene.json").is_file()
                   and seed_dir(frames_root, int(p.name[5:])).is_dir())
    todo = [s for s in seeds if not (seed_dir(frames_root, s) / "Depth").is_dir()]
    print(f"{chunk.name}: {len(seeds)} walls, {len(todo)} without depth", flush=True)
    t0, failed = time.time(), {}
    if todo:
        scratch.mkdir(exist_ok=True)
        parts = [todo[k::args.workers] for k in range(args.workers)]
        with ThreadPoolExecutor(args.workers) as pool:
            jobs = [pool.submit(rebuild, args.blender_py, args.build_flags, part, scratch,
                                scratch / f"rebuild_{k}.log") for k, part in enumerate(parts) if part]
            for job in jobs:
                job.result()
        for s in todo:
            new_frames = seed_dir(scratch / "frames", s)
            reason = check(seed_dir(walls, s), seed_dir(scratch, s), seed_dir(frames_root, s), new_frames)
            if reason:
                failed[s] = reason
                continue
            for folder, _, _ in FOLDERS:
                os.rename(new_frames / folder, seed_dir(frames_root, s) / folder)
    linked = link_pack(chunk)
    with_depth = sum((seed_dir(frames_root, s) / "Depth").is_dir() for s in seeds)
    git = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()
    record = {"walls": len(seeds), "walls_with_depth": with_depth, "failed": failed,
              "pack_files": {out: len(list((chunk / "pack" / out).glob(f"*{ext}"))) for _, out, ext in FOLDERS},
              "pack_images": len(list((chunk / "pack/images").glob("*.png"))),
              "minutes_this_run": round((time.time() - t0) / 60, 1),
              "code_commit": git("rev-parse", "HEAD"),
              "code_uncommitted_changes": bool(git("status", "--porcelain", "--", "syncrack3d", "cwg_infinigen")),
              "method": __doc__.split("\n\n")[1].replace("\n", " ")}
    (chunk / "depth_added.json").write_text(json.dumps(record, indent=1) + "\n")
    if not failed and scratch.exists():
        shutil.rmtree(scratch)
    print(f"{chunk.name}: {with_depth}/{len(seeds)} walls with depth, {len(failed)} failed, "
          f"{linked} pack links added, {record['minutes_this_run']} min", flush=True)
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--chunks", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--build-flags", default="--recipe unified",
                    help="the flags the walls were built with (their own recipe and options)")
    ap.add_argument("--blender-py", default=sys.executable)
    args = ap.parse_args()
    results = {name: add_chunk(args.dataset / name, args) for name in args.chunks}
    bad = {k: v["failed"] for k, v in results.items() if v["failed"]}
    print("DEPTH_ADD_DONE" if not bad else f"DEPTH_ADD_DONE with failures: {bad}", flush=True)


if __name__ == "__main__":
    main()
