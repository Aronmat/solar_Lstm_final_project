from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def cubic_spline_fill(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays.")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    valid = np.isfinite(y)
    if valid.sum() < 3:
        raise ValueError("At least 3 valid points are required for cubic spline interpolation.")

    spline = CubicSpline(x[valid], y[valid], extrapolate=True)
    y_filled = y.copy()
    y_filled[~valid] = spline(x[~valid])

    return y_filled