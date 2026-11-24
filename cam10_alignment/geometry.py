"""Perspective-aware mask geometry and robust camera-space plane estimation."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class GeometryError(ValueError):
    reason: str

    def __str__(self):
        return self.reason


def fit_plane(points, threshold=0.005, min_ratio=0.75, iterations=300, max_points=12000, seed=0):
    """RANSAC + SVD. Distances are metres, provided the input is metric."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    p = p[np.isfinite(p).all(axis=1) & (p[:, 2] > 0)]
    if threshold <= 0 or not 0 < min_ratio <= 1 or iterations < 1 or max_points < 3:
        raise GeometryError("invalid plane-fit parameters")
    if len(p) < 30:
        raise GeometryError("fewer than 30 finite positive-depth points")
    rng = np.random.default_rng(seed)
    sample = p[rng.choice(len(p), min(len(p), max_points), replace=False)]
    best = np.zeros(len(sample), dtype=bool)
    for _ in range(iterations):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(b - a, c - a)
        length = np.linalg.norm(n)
        if length < 1e-10:
            continue
        n /= length
        inside = np.abs((sample - a) @ n) <= threshold
        if inside.sum() > best.sum():
            best = inside
    if best.sum() < 30:
        raise GeometryError("no non-degenerate plane consensus")
    for _ in range(3):
        center = sample[best].mean(axis=0)
        _, s, vh = np.linalg.svd(sample[best] - center, full_matrices=False)
        if s[1] < 1e-6 or s[1] / max(s[0], 1e-12) < 1e-3:
            raise GeometryError("points are nearly collinear")
        n = vh[-1]
        best = np.abs((sample - center) @ n) <= threshold
        if best.sum() < 30:
            raise GeometryError("plane refinement lost consensus")
    # Refit once to final sample consensus, then assess all observations.
    center = sample[best].mean(axis=0)
    _, _, vh = np.linalg.svd(sample[best] - center, full_matrices=False)
    n = vh[-1]
    if n @ center > 0:
        n = -n  # visible face normal points towards the camera
    d = -n @ center
    residual = np.abs(p @ n + d)
    inside = residual <= threshold
    ratio = float(inside.mean())
    if ratio < min_ratio:
        raise GeometryError(f"plane inlier ratio {ratio:.3f} below {min_ratio}")
    return {"normal_camera": n.tolist(), "d_m": float(d),
            "inlier_centroid_camera_m": center.tolist(), "inlier_ratio": ratio,
            "inlier_rmse_m": float(np.sqrt(np.mean(residual[inside] ** 2))),
            "point_count": len(p), "inlier_count": int(inside.sum()),
            "ransac_threshold_m": threshold}, inside

