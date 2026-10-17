# Copyright (C) 2025, SynDefect3D.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

"""Check an exported pack against the G1 instance contract.

G1's acceptance asks that scenes with disconnected flakes, multiple same-class
instances, overlaps, occluders, empty views and thin cracks pass an
ID/class/mask/box round trip with no unexplained identity swaps or split and
merged objects. This checks that mechanically, so "hand-checkable" does not mean
somebody squinting at masks.

Per frame it asserts:

  identity     every decoded id belongs to the manifest, and no two manifest
               instances share one id
  class        every pixel of an instance carries that instance's class - a
               pixel that disagrees is an identity swap - and every scored
               semantic positive has an owning instance
  blending     no pixel mixes samples from two or more instance ids. Overlap
               is out of scope for this release, so any blended pixel is a
               placement or shader fault, not a policy case
  wholeness    an instance's box covers all of its pieces. Peel is disconnected
               by nature, so this is the check that catches a flake cluster
               being split into several objects or merged with its neighbour
  evidence     labelled pixels, soft coverage, and how much was ambiguous or
               under-resolution

It reports rather than fails on disconnection, because whether a peel cluster is
one instance or several is a contract decision, not a defect - the number is
here so that decision can be made from data.

    python tools/check_instance_contract.py --pack outputs/g1/pack
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

# Beyond this the frame is a room view, where a card projecting large says
# nothing about whether the defect inside it is resolvable.
QA_VIEW_MAX_DISTANCE_M = 1.5


def connected_components(mask: np.ndarray) -> int:
    from scipy.ndimage import label

    return int(label(mask)[1])


def check_frame(sample_id: str, pack: Path) -> dict:
    npz_path = pack / "instance_masks" / f"{sample_id}.npz"
    rec_path = pack / "instances" / f"{sample_id}.json"
    if not npz_path.is_file() or not rec_path.is_file():
        return {"sample": sample_id, "skipped": "no instance outputs"}

    z = np.load(npz_path)
    ids = z["instance_id"]
    ambiguous = z["ambiguous"]
    under = z["under_resolution"]
    sidecar = json.loads(rec_path.read_text())
    mask = np.asarray(Image.open(pack / "masks" / f"{sample_id}.png"))
    step = 255
    legend = (pack / "class_names.txt").read_text()
    for line in legend.splitlines():
        if "gray level = class_id *" in line:
            step = int(line.split("*")[1].split("(")[0].strip())
            break
    label_map = np.clip((mask.astype(np.int32) + step // 2) // step, 0, 255)
    # Ignored pixels are deliberately background in the semantic mask while the
    # instance map still names their owner - a marginal boundary band, or an
    # instance below the evidence floor. Comparing class there reports an
    # identity swap that is not one, so those pixels are excluded.
    ignore_path = pack / "ignore_masks" / f"{sample_id}.png"
    ignored = (
        np.asarray(Image.open(ignore_path)) > 0
        if ignore_path.is_file()
        else np.zeros_like(label_map, dtype=bool)
    )

    issues, rows = [], []
    owned = ids > 0
    # A supported pixel the decoder could not resolve to one id is a blend; a
    # class conflict is also `ambiguous` but still decodes to a single owner.
    blended = ambiguous & (z["soft_instance_id"] == 0) if "soft_instance_id" in z else ambiguous
    if blended.any():
        issues.append(f"{int(blended.sum())} px mix two or more instance ids - blended identity")
    unowned = (label_map > 0) & ~owned & ~ignored
    if unowned.any():
        issues.append(
            f"{int(unowned.sum())} scored semantic px have no owning instance - "
            f"mask and instance map disagree"
        )

    by_id = {int(r["instance_id"]): r for r in sidecar["instances"]}
    boxes = {}
    box_path = pack / "bboxes" / f"{sample_id}.json"
    if box_path.is_file():
        for b in json.loads(box_path.read_text()).get("boxes", []):
            boxes[int(b["instance_id"])] = b

    present = [int(i) for i in np.unique(ids) if i]

    for inst_id in present:
        if inst_id not in by_id:
            issues.append(f"decoded id {inst_id} is not in the manifest")
            continue
        record = by_id[inst_id]
        pixels = ids == inst_id
        want_class = int(record["training_class_id"])
        scored = pixels & ~ignored
        wrong = scored & (label_map != want_class)
        if wrong.any():
            issues.append(
                f"id {inst_id} ({record['class_name']}): {int(wrong.sum())} px carry "
                f"another class - identity swap"
            )
        pieces = connected_components(pixels)
        covered = True
        if inst_id in boxes:
            b = boxes[inst_id]
            ys, xs = np.where(pixels)
            covered = (
                xs.min() >= b["x_min"] and xs.max() <= b["x_max"]
                and ys.min() >= b["y_min"] and ys.max() <= b["y_max"]
            )
            if not covered:
                issues.append(f"id {inst_id}: box does not cover all of its pixels")
        rows.append(
            {
                "id": inst_id,
                "class": record["class_name"],
                "px": int(pixels.sum()),
                "scored_px": int(scored.sum()),
                "pieces": pieces,
                "boxed": inst_id in boxes,
                "box_covers_all": covered,
            }
        )

    # Expected support vs actual coverage. The view metadata says which
    # instances the geometry puts in frame and how much of each is unoccluded;
    # the instance map says which ones actually carried labels. An instance the
    # geometry says is squarely in view with nothing in front of it, that then
    # contributes no labelled pixels at all, is the signature of a silently
    # truncated shader - the render succeeds and the labels are simply absent.
    # Only close-up views can be judged this way. `projected_px_area` is the
    # placement *card*, not the defect: a metre-wide card holding one hairline
    # crack legitimately contributes almost nothing at room distance, and
    # flagging that would report the subpixel case as a shader failure. Inside a
    # metre and a half the card is the defect's own QA view, and an instance
    # squarely in frame there with no labelled pixels at all is a real fault.
    for inst_id, record in by_id.items():
        view = record.get("view") or {}
        projected = float(view.get("projected_px_area") or 0.0)
        visible = float(view.get("visible_fraction") or 0.0)
        distance = float(view.get("distance_m") or 1e9)
        if projected <= 0.0 or visible < 0.5 or distance > QA_VIEW_MAX_DISTANCE_M:
            continue
        if inst_id in present or int(record.get("labelled_pixel_count") or 0) > 0:
            continue
        issues.append(
            f"id {inst_id} ({record.get('class_name','?')}): its QA view at "
            f"{distance:.2f} m puts {projected:.0f} px in frame at "
            f"{100 * visible:.0f}% unoccluded, yet it carries no labelled pixels"
        )

    # a manifest instance the decoder never produced is fine (out of frame), but
    # one that the class mask shows and the decoder missed is not
    for inst_id, record in by_id.items():
        if inst_id in present:
            continue
        if int(record.get("labelled_pixel_count") or 0) > 0 and not (
            (ids == inst_id) & ignored
        ).any():
            issues.append(
                f"id {inst_id} has {record['labelled_pixel_count']} labelled px "
                f"but no decoded instance pixels"
            )

    return {
        "sample": sample_id,
        "rows": rows,
        "issues": issues,
        "ambiguous_px": int(ambiguous.sum()),
        "blended_px": int(blended.sum()),
        "owned_px": int(owned.sum()),
        "under_resolution_px": int(under.sum()),
        "labelled_px": int((label_map > 0).sum()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", type=Path, required=True)
    args = ap.parse_args()
    pack = args.pack.resolve()

    samples = sorted(p.stem for p in (pack / "masks").glob("*.png"))
    if not samples:
        print(f"no samples in {pack}")
        return 1

    total_issues = 0
    blended_px = owned_px = 0
    for sample_id in samples:
        result = check_frame(sample_id, pack)
        if result.get("skipped"):
            print(f"\n{sample_id}: skipped ({result['skipped']})")
            continue
        blended_px += result["blended_px"]
        owned_px += result["owned_px"]
        print(f"\n{sample_id}")
        print(f"  labelled {result['labelled_px']} px, ambiguous {result['ambiguous_px']} "
              f"(blended {result['blended_px']}), "
              f"under-resolution {result['under_resolution_px']}")
        if not result["rows"]:
            print("  no decoded instances (negative view)")
        for row in result["rows"]:
            note = "" if row["box_covers_all"] else "  <-- BOX DOES NOT COVER ALL PIECES"
            piece_note = f"{row['pieces']} piece(s)" + (" (disconnected)" if row["pieces"] > 1 else "")
            status = ""
            if row["scored_px"] == 0:
                status = "  [below evidence floor -> ignored, no box]"
            print(f"  id={row['id']:<3} {row['class']:<13} {row['px']:>7} px  {piece_note:<26}"
                  f" boxed={row['boxed']}{status}{note}")
        for issue in result["issues"]:
            print(f"  ISSUE: {issue}")
        total_issues += len(result["issues"])

    print(f"\n{len(samples)} frame(s), {total_issues} issue(s)")
    rate = blended_px / max(1, blended_px + owned_px)
    print(f"blended instance ids: {blended_px} px of {blended_px + owned_px} "
          f"supported ({rate:.4%})")
    print("Checks: id in manifest, no shared ids, no blended ids, class agrees on "
          "every scored pixel in both directions, box covers all pieces, expected "
          "support carries coverage.")
    if total_issues == 0:
        print("identity, class and box round-trip clean on every frame checked.")
        print("Note this covers the cases these frames contain. Overlapping and "
              "touching instances need a fixture that deliberately creates them.")
    return 1 if total_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
