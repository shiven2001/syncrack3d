"""Width-insensitive crack metrics: centreline precision/recall/F1 and clDice.

Same definition as domain_transfer/crack_metrics.py. Pixel IoU rewards matching
an annotator's polygon width as well as finding the crack. Centreline metrics
thin both masks to 1-px skeletons: recall is the share of labelled crack length
within `tol` px of the prediction, precision the share of predicted length
within `tol` px of the label. clDice (Shit et al., CVPR 2021) uses
skeleton-inside-mask without a tolerance. Per-photo functions return counts so
a benchmark pools them by summation (length-weighted), like pooled IoU.
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize


def _grow(mask: np.ndarray, tol: int) -> np.ndarray:
    if tol == 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def centreline_counts(pred: np.ndarray, gt: np.ndarray, tol: int) -> np.ndarray:
    """[gt length hit, gt length, pred length hit, pred length] for boolean masks."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    ps, gs = skeletonize(pred), skeletonize(gt)
    return np.array([(gs & _grow(pred, tol)).sum(), gs.sum(), (ps & _grow(gt, tol)).sum(), ps.sum()],
                    dtype=np.int64)


def cldice_counts(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """[pred skeleton in gt, pred skeleton, gt skeleton in pred, gt skeleton]."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    ps, gs = skeletonize(pred), skeletonize(gt)
    return np.array([(ps & gt).sum(), ps.sum(), (gs & pred).sum(), gs.sum()], dtype=np.int64)


def centreline_scores(c: np.ndarray) -> dict:
    """Pooled precision/recall/F1 from summed centreline_counts."""
    recall = float(c[0] / c[1]) if c[1] else float("nan")
    precision = float(c[2] / c[3]) if c[3] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
