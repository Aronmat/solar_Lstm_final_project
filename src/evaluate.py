import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    d = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return float(np.mean(np.where(d > 1e-6, np.abs(y_true - y_pred) / d, 0.0)) * 100.0)


def mape_thresh(y_true, y_pred, min_denom=10.0):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = np.abs(y_true) >= min_denom

    if not np.any(mask):
        return float("nan")

    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)