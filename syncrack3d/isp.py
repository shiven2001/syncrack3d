"""Optional RGB-only camera processing; raw RGB and all labels are preserved."""

import cv2
import numpy as np


def process_rgb(image, seed):
    rng = np.random.default_rng(seed)
    sigma = float(rng.uniform(1.0, 3.0))
    quality = int(rng.integers(88, 99))
    blur = float(rng.uniform(0.0, 0.45))
    rgb = image.astype(np.float32)
    if blur > 0.3:
        rgb = cv2.GaussianBlur(rgb, (3, 3), blur)
    # Luminance-dominant noise avoids strongly colored speckle on neutral paint.
    rgb += rng.normal(0, sigma, image.shape[:2])[..., None]
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR), {
        "seed": int(seed),
        "noise_sigma": sigma,
        "jpeg_quality": quality,
        "blur_sigma": blur,
    }


def process_frames(frames_dir, stems, seed):
    records = []
    for i, stem in enumerate(stems):
        path = frames_dir / "Image" / "camera_0" / f"Image_{stem}.png"
        raw = frames_dir / "ImageRaw" / "camera_0" / path.name
        if raw.exists():
            raise FileExistsError(f"Refusing to process RGB twice: {raw}")
        im = cv2.imread(str(path))
        if im is None:
            raise FileNotFoundError(path)
        processed, metadata = process_rgb(im, (int(seed) * 1009 + i) % (2**32))
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(path.read_bytes())
        if not cv2.imwrite(str(path), processed):
            raise RuntimeError(f"Cannot write {path}")
        records.append(dict(stem=stem, **metadata))
    return records
