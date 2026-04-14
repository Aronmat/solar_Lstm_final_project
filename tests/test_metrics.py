import numpy as np
import pytest
from src.evaluate import rmse


def test_rmse_known_values():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])

    result = rmse(y_true, y_pred)

    assert result == pytest.approx((1.0 / 3.0) ** 0.5)


def test_rmse_raises_for_shape_mismatch():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0])

    try:
        rmse(y_true, y_pred)
        assert False, "Expected ValueError"
    except ValueError:
        assert True