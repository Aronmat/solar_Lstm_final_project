import numpy as np
from src.spline_interpolation import cubic_spline_fill


def test_cubic_spline_fill_removes_nans():
    x = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    y = np.array([1.0, np.nan, 2.0, np.nan, 4.0, 5.0], dtype=float)

    result = cubic_spline_fill(x, y)

    assert len(result) == len(y)
    assert not np.isnan(result).any()


def test_cubic_spline_fill_raises_with_too_few_points():
    x = np.array([0, 1, 2], dtype=float)
    y = np.array([1.0, np.nan, np.nan], dtype=float)

    try:
        cubic_spline_fill(x, y)
        assert False, "Expected ValueError"
    except ValueError:
        assert True