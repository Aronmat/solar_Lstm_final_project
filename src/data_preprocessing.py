import numpy as np
import pandas as pd
from pathlib import Path

from src.spline_interpolation import cubic_spline_fill


def load_historical_dataset(path: Path, target: str, feature_cols, use_spline=True):
    if not path.exists():
        raise FileNotFoundError(f"Missing historical dataset file: {path}")

    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    return clean_dataset(df, target, feature_cols, use_spline=use_spline)


def load_local_dataset(folder: Path, pattern: str, target: str, feature_cols, use_spline=True):
    files = sorted(folder.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No local CSV files found in {folder} matching {pattern}")

    frames = []
    for file in files:
        tmp = pd.read_csv(file)
        tmp["source_file"] = file.name
        frames.append(tmp)

    df = pd.concat(frames, ignore_index=True)

    return clean_dataset(df, target, feature_cols, use_spline=use_spline)


def clean_dataset(df, target: str, feature_cols, use_spline=True):
    if "utc_timestamp" in df.columns and "timestamp" not in df.columns:
        df.rename(columns={"utc_timestamp": "timestamp"}, inplace=True)

    if "timestamp" not in df.columns:
        raise ValueError("Dataset must contain a 'timestamp' or 'utc_timestamp' column.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df.dropna(subset=["timestamp"], inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if target not in df.columns:
        raise ValueError(f"'{target}' must exist in dataset.")

    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    df[target] = pd.to_numeric(df[target], errors="coerce")

    if use_spline:
        x = np.arange(len(df), dtype=float)
        if np.isfinite(df[target].values).sum() >= 3:
            df[target] = cubic_spline_fill(x, df[target].values)
        else:
            df[target] = df[target].ffill().bfill()

    df[target] = df[target].rolling(5, min_periods=1).mean()

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

        if df[col].isna().any():
            x = np.arange(len(df), dtype=float)
            values = df[col].values

            if use_spline and np.isfinite(values).sum() >= 3:
                df[col] = cubic_spline_fill(x, values)
            else:
                df[col] = df[col].ffill().bfill()

    df.dropna(subset=feature_cols + [target], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df, feature_cols


def make_normalizers(train_df, target):
    ylog = np.log1p(np.clip(train_df[target].values, 0, None))
    mean_log = float(np.mean(ylog))
    std_log = float(np.std(ylog) or 1.0)

    def to_norm(y):
        return (np.log1p(np.clip(y, 0, None)) - mean_log) / std_log

    def from_norm(z):
        return np.maximum(np.expm1(z * std_log + mean_log), 0.0)

    return to_norm, from_norm


def make_sequences_with_target(df, feature_cols, target_col, steps, to_norm, forecast_horizon=1):
    V = df[feature_cols].values.astype(np.float32)
    T = df[target_col].values.astype(np.float32)
    Tn = to_norm(T)

    X, Y = [], []

    max_i = len(df) - steps - forecast_horizon + 1

    for i in range(max_i):
        x_features = V[i:i + steps]
        x_target = Tn[i:i + steps].reshape(steps, 1)
        y_future = Tn[i + steps + forecast_horizon - 1]

        X.append(np.concatenate([x_features, x_target], axis=1))
        Y.append([y_future])

    return np.asarray(X, np.float32), np.asarray(Y, np.float32)


def time_ordered_split(X, Y, train_frac=0.80, val_frac=0.10):
    s1 = int(train_frac * len(X))
    s2 = int((train_frac + val_frac) * len(X))

    return X[:s1], Y[:s1], X[s1:s2], Y[s1:s2], X[s2:], Y[s2:]