"""Write a score.py manifest for one rendered split (walls/ + pack/).

    python eval/synth/manifest.py SPLIT_DIR

SPLIT_DIR holds build_dataset.py output in walls/ and its export in pack/.
Views are grouped by the crack style each wall drew (standard, wide, crazing;
scene.json appearance_controls.crack_style), so score.py reports per
style, per material and pooled. Writes SPLIT_DIR/manifest.json.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2

APP = Path(__file__).resolve().parents[2]
root = Path(sys.argv[1]).resolve()
scenes, views = {}, []
for img in sorted((root / "pack" / "images").glob("*.png")):
    seed = int(img.name.split("_")[1])
    if seed not in scenes:
        scenes[seed] = json.loads((root / "walls" / f"seed_{seed}" / "scene.json").read_text())
    rec = scenes[seed]
    mask = root / "pack" / "masks" / img.name
    views.append(dict(
        subset=rec["appearance_controls"].get("crack_style") or "unstyled", id=img.stem, seed=seed,
        image=str(img.relative_to(root)), mask=str(mask.relative_to(root)),
        material=rec["wall_material"].get("diverse", {}).get("category", rec["wall_material"].get("generator")),
        environment=rec.get("environment"), crack_px=int((cv2.imread(str(mask), 0) == 60).sum()),
        sha256_image=hashlib.sha256(img.read_bytes()).hexdigest()))
commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=APP, capture_output=True, text=True).stdout.strip()
summary = {}
for v in views:
    s = summary.setdefault(v["subset"], {"views": 0, "walls": set(), "cracked_views": 0})
    s["views"] += 1
    s["walls"].add(v["seed"])
    s["cracked_views"] += v["crack_px"] > 0
summary = {k: {**s, "walls": len(s["walls"])} for k, s in summary.items()}
(root / "manifest.json").write_text(json.dumps(dict(name=root.name, code_commit=commit, summary=summary,
                                                    views=views), indent=1))
print(root.name, json.dumps(summary))
