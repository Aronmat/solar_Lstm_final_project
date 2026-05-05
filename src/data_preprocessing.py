import numpy as np
import pandas as pd
from pathlib import Path

from src.spline_interpolation import cubic_spline_fill


def load_pair(tp: Path, pp: Path, target: str, use_spline=True):
    if not tp.exists():
        raise FileNotFoundError(f"Missing trainer CSV: {tp}")
    if not pp.exists():
        raise FileNotFoundError(f"Missing predictor CSV: {pp}")

    train_df = pd.read_csv(tp)
    pred_df = pd.read_csv(pp)

    for df in (train_df, pred_df):
        if "utc_timestamp" in df.columns and "timestamp" not in df.columns:
            df.rename(columns={"utc_timestamp": "timestamp"}, inplace=True)

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df.dropna(subset=["timestamp"], inplace=True)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        if target not in df.columns:
            raise ValueError(f"'{target}' must exist in both CSVs.")

        df[target] = pd.to_numeric(df[target], errors="coerce")

        if use_spline:
            x = np.arange(len(df), dtype=float)
            df[target] = cubic_spline_fill(x, df[target].values)

        df[target] = df[target].rolling(5, min_periods=1).mean()

    def pc_key(col):
        s = str(col)
        return int("".join(ch for ch in s if ch.isdigit()) or 0)

    pcs_t = [c for c in train_df.columns if str(c).upper().startswith("PC")]
    pcs_p = [c for c in pred_df.columns if str(c).upper().startswith("PC")]

    pc_cols = [c for c in sorted(pcs_t, key=pc_key) if c in pcs_p]

    if not pc_cols:
        raise ValueError("No common PC columns between trainer and predictor.")

    return train_df, pred_df, pc_cols


def make_normalizers(train_df, target):
    ylog = np.log1p(np.clip(train_df[target].values, 0, None))
    mean_log = float(np.mean(ylog))
    std_log = float(np.std(ylog) or 1.0)

    def to_norm(y):
        return (np.log1p(np.clip(y, 0, None)) - mean_log) / std_log

    def from_norm(z):
        return np.maximum(np.expm1(z * std_log + mean_log), 0.0)

    return to_norm, from_norm


def make_sequences_with_target(df, pc_cols, target_col, steps, to_norm):
    V = df[pc_cols].values.astype(np.float32)
    T = df[target_col].values.astype(np.float32)
    Tn = to_norm(T)

    X, Y = [], []

    for i in range(len(df) - steps):
        x_pc = V[i:i + steps]
        x_target = Tn[i:i + steps].reshape(steps, 1)
        y_next = Tn[i + steps]

        X.append(np.concatenate([x_pc, x_target], axis=1))
        Y.append([y_next])

    return np.asarray(X, np.float32), np.asarray(Y, np.float32)


def time_ordered_split(X, Y, train_frac=0.70, val_frac=0.15):
    s1 = int(train_frac * len(X))
    s2 = int((train_frac + val_frac) * len(X))

    return X[:s1], Y[:s1], X[s1:s2], Y[s1:s2], X[s2:], Y[s2:]