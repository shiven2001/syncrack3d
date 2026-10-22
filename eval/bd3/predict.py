"""Write BD3 class-map predictions for every checkpoint seed (protocol v2: 1.5x upscaling).

    python eval/bd3/predict.py --model path/to/checkpoint_dir [--scale 1.5] [--out-name pred_v2]

For each runs/s*/last.pt under --model, predicts the 842 photos in manifest.json and
writes <model>/<out-name>/s*/<id>.png (crack = 1), plus protocol.json recording the
scale. Score each seed with score.py --pred <model>/<out-name>/s<k>. Existing complete
seed folders are kept, so a crash resumes.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infer import TEST_SCALE, load_model, predict  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True, help="directory containing runs/s*/last.pt")
    ap.add_argument("--scale", type=float, default=TEST_SCALE, help="test-time upscaling (v1: 1)")
    ap.add_argument("--out-name", default="pred_v2")
    args = ap.parse_args()
    import torch

    photos = json.loads((HERE / "manifest.json").read_text())["photos"]
    seeds = sorted(args.model.glob("runs/s*/last.pt"))
    if not seeds:
        raise SystemExit(f"no runs/s*/last.pt under {args.model}")
    for ck in seeds:
        dest = args.model / args.out_name / ck.parent.name
        proto = dest / "protocol.json"
        if proto.exists() and json.loads(proto.read_text()).get("test_scale") == args.scale \
                and all((dest / f"{p['id']}.png").exists() for p in photos):
            continue
        dest.mkdir(parents=True, exist_ok=True)
        model = load_model(ck)
        for p in photos:
            img = cv2.cvtColor(cv2.imread(str(HERE / p["image"])), cv2.COLOR_BGR2RGB)
            cv2.imwrite(str(dest / f"{p['id']}.png"), predict(model, img, args.scale))
        proto.write_text(json.dumps({"test_scale": args.scale, "checkpoint": str(ck)}) + "\n")
        del model
        torch.cuda.empty_cache()
        print("predicted", ck.parent.name, flush=True)


if __name__ == "__main__":
    main()
