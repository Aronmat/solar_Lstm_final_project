import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ========= EDIT THESE TWO PATHS IF NEEDED =========
INPUT_PATH  = r"C:\Users\dchoa\Documents\spyder code files\solar_energy_data.csv"
OUTPUT_DIR  = r"C:\Users\dchoa\Documents\spyder code files\lstm files finalization"
# ==================================================

OUT_TRAIN = "pca_trainer_with_power_output_HALFYEAR.csv"
OUT_PRED  = "pca_predictor_cleaned_matched_HALFYEAR.csv"

# Which raw features to use if present (use whatever exists from this list)
CANDIDATE_FEATURES = ["temperature", "humidity", "dew", "wind_speed", "cloud_coverage"]

def main():
    in_path = Path(INPUT_PATH)
    if not in_path.exists():
        raise FileNotFoundError(f"Could not find input file:\n{INPUT_PATH}")

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    df.columns = [c.strip().lower() for c in df.columns]

    # normalize/parse timestamp
    if "utc_timestamp" in df.columns and "timestamp" not in df.columns:
        df.rename(columns={"utc_timestamp": "timestamp"}, inplace=True)
    if "timestamp" not in df.columns:
        raise ValueError("Expected a 'timestamp' column in solar_energy_data.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if "power_output" not in df.columns:
        raise ValueError("Expected a 'power_output' column in solar_energy_data.csv")

    # choose features that actually exist
    features = [c for c in CANDIDATE_FEATURES if c in df.columns]
    if not features:
        raise ValueError(f"No expected weather features found. Looked for: {CANDIDATE_FEATURES}")

    # split by first vs second 6 months starting from dataset min time
    start = df["timestamp"].min()
    mid   = start + pd.DateOffset(months=6)
    train_df = df[df["timestamp"] < mid].copy()
    pred_df  = df[df["timestamp"] >= mid].copy()

    if len(train_df) < 100 or len(pred_df) < 100:
        print("Warning: one of the halves has very few rows; proceed with caution.")

    # Scale + PCA on trainer; apply to predictor
    scaler = StandardScaler()
    X_train = train_df[features].copy().ffill().bfill().fillna(0.0)
    X_train_scaled = scaler.fit_transform(X_train)

    n_components = min(5, X_train_scaled.shape[1])  # 2..5 depending on available features
    pca = PCA(n_components=n_components, random_state=42)
    pcs_train = pca.fit_transform(X_train_scaled)
    pc_cols = [f"PC{i}" for i in range(1, n_components + 1)]

    # compose trainer output
    trainer_out_df = pd.concat(
        [
            train_df[["timestamp"]].reset_index(drop=True),
            pd.DataFrame(pcs_train, columns=pc_cols),
            train_df[["power_output"]].reset_index(drop=True),
        ],
        axis=1,
    )
    trainer_out_path = out_dir / OUT_TRAIN
    trainer_out_df.to_csv(trainer_out_path, index=False)

    # predictor transform
    X_pred = pred_df[features].copy().ffill().bfill().fillna(0.0)
    X_pred_scaled = scaler.transform(X_pred)
    pcs_pred = pca.transform(X_pred_scaled)

    predictor_out_df = pd.concat(
        [
            pred_df[["timestamp"]].reset_index(drop=True),
            pd.DataFrame(pcs_pred, columns=pc_cols),
            pred_df[["power_output"]].reset_index(drop=True),
        ],
        axis=1,
    )
    predictor_out_path = out_dir / OUT_PRED
    predictor_out_df.to_csv(predictor_out_path, index=False)

    # helpful prints
    print("\n=== PCA Half-Year Split Complete ===")
    print(f"Features used: {features}")
    print(f"PCs: {pc_cols}")
    print(f"Explained variance ratio: {np.round(pca.explained_variance_ratio_, 4).tolist()}")
    print(f"\nTrainer saved to:   {trainer_out_path}")
    print(f"Predictor saved to: {predictor_out_path}")

if __name__ == "__main__":
    main()
