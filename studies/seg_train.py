"""3-class crack segmentation training for the studies (U-Net R34, DeepLabV3+ R50, SegFormer-B2).

Labels: 0 background, 1 crack, 2 second class (unused by the crack scorers),
255 ignore. Loss cross-entropy + soft Dice over classes 1-2, non-ignore pixels;
AdamW, 200 warm-up steps then cosine; batch 8 at 512 px crops; random scale
0.75-1.25, 90-degree rotations, flips, brightness/contrast/saturation jitter;
the final checkpoint is kept (no validation-based selection). Synthetic rows
(real flag False) can have their crack labels grown by `syn_envelope` px and
get a simulated phone camera (gamma, gain, blur, noise, JPEG). Checkpoints load
with eval/infer.py, which reads the arch.json written beside them.
"""

import json
import math
import random
import time

import cv2
import numpy as np

IGNORE = 255
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255
LAZY_ABOVE = 3000
ARCHS = {  # same names as eval/infer.py
    "unet_r34": ("Unet", "resnet34"),
    "deeplabv3plus_r50": ("DeepLabV3Plus", "resnet50"),
    "segformer_b2": ("Segformer", "mit_b2"),
}


def load_item(img_path, mask_path, real):
    """RGB image and label map; a real row's mask is binarised at 127, a synthetic one is read as is."""
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    lab = (m > 127).astype(np.uint8) if real else m
    return img, lab


def class_envelope(lab, radius):
    """Grow each class by `radius` px into background and ignore; crack wins where both grow."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    out = lab.copy()
    free = (lab == 0) | (lab == IGNORE)
    for c in (2, 1):
        grown = (cv2.dilate((lab == c).astype(np.uint8), kernel) > 0) & free
        out[grown] = c
    return out


def camera_variation(image, rng):
    """Photometric randomisation of a synthetic RGB image: gamma, gain, blur, noise, JPEG."""
    rgb = image.astype(np.float32) / 255
    gamma = rng.uniform(0.55, 1.35)
    gain = rng.uniform(0.8, 1.6)
    rgb = np.clip(gain * np.power(rgb, gamma), 0, 1)
    if rng.random() < 0.5:
        rgb = cv2.GaussianBlur(rgb, (3, 3), rng.uniform(0.3, 0.9))
    sigma = rng.uniform(0, 0.018)
    rgb += rng.normal(0, sigma, rgb.shape[:2])[..., None]
    rgb = np.uint8(np.clip(rgb * 255, 0, 255))
    # The codec treats RGB as an uninterpreted 3-channel array; encode/decode keeps the order.
    ok, encoded = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(65, 96))])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


class SegSet:
    """Rows (name, image path, label path, real) -> augmented 512 px crops, normalised for ImageNet encoders."""

    def __init__(self, items, imgsz, seed, syn_envelope=0, syn_camera=False):
        self.items = items
        self.synthetic = [not r for _, _, _, r in items]
        self.syn_envelope = syn_envelope
        # Large sets (the scaling curve: 19k renders ~ 30 GB decoded) are read per access.
        self.lazy = len(items) > LAZY_ABOVE
        self.data = None if self.lazy else [self._load(k) for k in range(len(items))]
        self.syn_camera = syn_camera
        self.imgsz = imgsz
        self.rng = np.random.default_rng(seed)

    def _load(self, k):
        _, i, m, r = self.items[k]
        im, lab = load_item(i, m, r)
        return im, class_envelope(lab, self.syn_envelope) if self.syn_envelope else lab

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        import torch

        rng = np.random.default_rng((self.rng.integers(1 << 31), idx))
        img, lab = self._load(idx) if self.lazy else self.data[idx]
        if self.syn_camera and self.synthetic[idx]:
            img = camera_variation(img, rng)  # a new array; the stored image is untouched
        s = rng.uniform(0.75, 1.25)
        h, w = lab.shape
        size = (max(1, round(w * s)), max(1, round(h * s)))
        img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        lab = cv2.resize(lab, size, interpolation=cv2.INTER_NEAREST)
        k = int(rng.integers(4))
        img, lab = np.rot90(img, k), np.rot90(lab, k)
        if rng.random() < 0.5:
            img, lab = img[:, ::-1], lab[:, ::-1]
        if rng.random() < 0.5:
            img, lab = img[::-1], lab[::-1]
        img = img.astype(np.float32)
        img = (img - img.mean()) * rng.uniform(0.8, 1.2) + img.mean() * rng.uniform(0.8, 1.2)
        gray = img.mean(axis=2, keepdims=True)
        img = np.clip(gray + (img - gray) * rng.uniform(0.8, 1.2), 0, 255)
        t = self.imgsz
        h, w = lab.shape
        ph, pw = max(0, t - h), max(0, t - w)
        if ph or pw:
            img = np.pad(img, ((0, ph), (0, pw), (0, 0)), constant_values=0)
            lab = np.pad(lab, ((0, ph), (0, pw)), constant_values=IGNORE)
        h, w = lab.shape
        y, x = int(rng.integers(h - t + 1)), int(rng.integers(w - t + 1))
        img, lab = img[y:y + t, x:x + t], lab[y:y + t, x:x + t]
        img = (img - MEAN) / STD
        return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))), torch.from_numpy(np.ascontiguousarray(lab))


def train(items, run_dir, seed, iters=4000, batch=8, imgsz=512, workers=4,
          syn_envelope=2, syn_camera=True, arch="unet_r34", lr=3e-4):
    """Train one seed on `items` and write run_dir/last.pt, arch.json and train.json."""
    import segmentation_models_pytorch as smp
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    ds = SegSet(items, imgsz, seed, syn_envelope=syn_envelope, syn_camera=syn_camera)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True, num_workers=workers,
                        persistent_workers=True, generator=torch.Generator().manual_seed(seed))
    cls, encoder = ARCHS[arch]
    model = getattr(smp, cls)(encoder, encoder_weights="imagenet", classes=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warm = 200
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: (i + 1) / warm if i < warm else
        0.5 * (1 + math.cos(math.pi * (i - warm) / max(1, iters - warm))))
    scaler = torch.amp.GradScaler()

    def loss_fn(logits, lab):
        lab = lab.long()
        ce = F.cross_entropy(logits, lab, ignore_index=IGNORE)
        valid = (lab != IGNORE).unsqueeze(1)
        prob = torch.softmax(logits, 1) * valid
        target = F.one_hot(lab.clamp(max=2), 3).permute(0, 3, 1, 2) * valid
        inter = (prob * target).sum((0, 2, 3))
        dice = 1 - (2 * inter + 1) / (prob.sum((0, 2, 3)) + target.sum((0, 2, 3)) + 1)
        return ce + dice[1:].mean()

    t0, it = time.time(), 0
    while it < iters:
        for x, y in loader:
            model.train()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
            loss = loss_fn(logits.float(), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at iteration {it}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            it += 1
            if it % 500 == 0:
                print(f"seed {seed} iter {it} loss {float(loss):.4f}", flush=True)
            if it >= iters:
                break
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "last.pt")
    (run_dir / "arch.json").write_text(json.dumps({"arch": arch, "encoder_weights": "imagenet"}) + "\n")
    (run_dir / "train.json").write_text(json.dumps(
        {"seed": seed, "iters": iters, "batch": batch, "imgsz": imgsz, "arch": arch, "lr": lr,
         "train_seconds": round(time.time() - t0), "n_train": len(ds)}, indent=1) + "\n")
    del model
    torch.cuda.empty_cache()
