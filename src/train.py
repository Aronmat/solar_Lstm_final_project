import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

from src.config import (
    HISTORICAL_DATA_PATH,
    LOCAL_DATA_DIR,
    LOCAL_FILE_PATTERN,
    HISTORICAL_TARGET,
    LOCAL_TARGET,
    HISTORICAL_FEATURE_COLS,
    LOCAL_FEATURE_COLS,
    HISTORICAL_TIMESTEPS,
    HISTORICAL_FORECAST_HORIZON,
    LOCAL_TIMESTEPS,
    LOCAL_FORECAST_HORIZON,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    SEED,
    DEVICE,
    USE_SPLINE,
    SHOW_BATCH_PROGRESS,
    BATCH_PRINT_EVERY,
    RUN_FEATURE_IMPORTANCE,
)

from src.data_preprocessing import (
    load_historical_dataset,
    load_local_dataset,
    make_normalizers,
    make_sequences_with_target,
    time_ordered_split,
)

from src.lstm_model import LSTMReg
from src.evaluate import rmse, mae, smape, mape_thresh
from src.plotting import plot_training_loss, plot_predictions, plot_daily_trend
from src.feature_importance import permutation_feature_importance


def choose_mode():
    print("\nSelect mode:")
    print("1 - Historical dataset")
    print("2 - Local Endeavor solar dataset")
    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        return "historical"
    if choice == "2":
        return "local"

    raise ValueError("Invalid choice. Enter 1 or 2.")


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    mode = choose_mode()

    if mode == "historical":
        print("\n--- Running Historical Dataset Mode ---")
        df, feature_cols = load_historical_dataset(
            HISTORICAL_DATA_PATH,
            HISTORICAL_TARGET,
            HISTORICAL_FEATURE_COLS,
            use_spline=USE_SPLINE,
        )
        
        target = HISTORICAL_TARGET
        timesteps = HISTORICAL_TIMESTEPS
        forecast_horizon = HISTORICAL_FORECAST_HORIZON
        dataset_label = str(HISTORICAL_DATA_PATH)

    else:
        print("\n--- Running Local Dataset Mode ---")

        df, feature_cols = load_local_dataset(
            LOCAL_DATA_DIR,
            LOCAL_FILE_PATTERN,
            LOCAL_TARGET,
            LOCAL_FEATURE_COLS,
            use_spline=USE_SPLINE,
        )

        target = LOCAL_TARGET
        timesteps = LOCAL_TIMESTEPS
        forecast_horizon = LOCAL_FORECAST_HORIZON
        dataset_label = str(LOCAL_DATA_DIR / LOCAL_FILE_PATTERN)

    print(f"[Solar LSTM] Dataset: {dataset_label}")
    print(f"[Solar LSTM] Using target: {target}")
    print(f"[Solar LSTM] Features: {feature_cols}")
    print(f"[Solar LSTM] Timesteps: {timesteps}")
    print(f"[Solar LSTM] Forecast horizon: {forecast_horizon}")
    print(f"[Solar LSTM] Cubic spline enabled: {USE_SPLINE}")

    to_norm, from_norm = make_normalizers(df, target)

    X_all, Y_all = make_sequences_with_target(
        df,
        feature_cols,
        target,
        timesteps,
        to_norm,
        forecast_horizon=forecast_horizon,
    )

    X_train, Y_train, X_val, Y_val, X_test, Y_test = time_ordered_split(
        X_all,
        Y_all,
        train_frac=0.80,
        val_frac=0.10,
    )

    print(f"Total rows: {len(df)}")
    print(f"Total sequences: {len(X_all)}")
    print(f"Train size: {len(X_train)}")
    print(f"Validation size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(Y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    print("[Solar LSTM] Alignment check: OK because split is time ordered.")

    input_size = len(feature_cols) + 1

    model = LSTMReg(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    criterion = nn.SmoothL1Loss(beta=1.0)

    opt = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=PATIENCE // 2,
    )

    best_vloss = float("inf")
    best_state = None
    no_improve = 0
    losses = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss += float(loss.item())

            if SHOW_BATCH_PROGRESS and batch_idx % BATCH_PRINT_EVERY == 0:
                print(
                    f"  Epoch {epoch:03d} | "
                    f"Batch {batch_idx}/{len(train_loader)} | "
                    f"Batch Loss: {loss.item():.4f}"
                )

        model.eval()

        with torch.no_grad():
            if len(X_val):
                val_pred = model(torch.tensor(X_val).to(DEVICE))
                vloss = criterion(val_pred, torch.tensor(Y_val).to(DEVICE)).item()
            else:
                vloss = epoch_loss

        sched.step(vloss)

        if epoch % 10 == 0:
            print(
                f"[Solar LSTM] Epoch {epoch:03d} | "
                f"TrainLoss {epoch_loss:.4f} | ValLoss {vloss:.4f}"
            )

        losses.append(epoch_loss)

        if vloss + 1e-6 < best_vloss:
            best_vloss = vloss
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1

            if no_improve >= PATIENCE:
                print(f"[Solar LSTM] Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    plot_training_loss(losses)

    model.eval()

    with torch.no_grad():
        yhat_norm = model(torch.tensor(X_test).to(DEVICE)).cpu().numpy().flatten()

    y_pred = from_norm(yhat_norm)
    y_true = from_norm(Y_test.flatten())
    y_pred = np.clip(y_pred, 0.0, None)

    test_start_index = timesteps + len(X_train) + len(X_val) + forecast_horizon - 1
    test_end_index = test_start_index + len(y_pred)

    ts = pd.to_datetime(df["timestamp"], utc=True).iloc[test_start_index:test_end_index]

    mask = np.isfinite(y_true) & np.isfinite(y_pred) & pd.Series(ts).notna().values
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    ts = ts.iloc[mask.nonzero()[0]]

    target_values = df[target].values.astype(float)

    naive = []
    true_baseline = []

    for i in range(test_start_index, test_end_index):
        if i - forecast_horizon >= 0 and i < len(target_values):
            true_baseline.append(target_values[i])
            naive.append(target_values[i - forecast_horizon])

    true_baseline = np.asarray(true_baseline)
    naive = np.asarray(naive)

    valid = np.isfinite(true_baseline) & np.isfinite(naive)
    true_baseline = true_baseline[valid]
    naive = naive[valid]

    print("\n=== Baseline vs Model on Test Set ===")

    print(
        "Persistence -> RMSE: %.2f | MAE: %.2f | SMAPE: %.2f%%"
        % (
            rmse(true_baseline, naive),
            mae(true_baseline, naive),
            smape(true_baseline, naive),
        )
    )

    print(
        "Model       -> RMSE: %.2f | MAE: %.2f | SMAPE: %.2f%% | MAPE@>=10: %.2f%%"
        % (
            rmse(y_true, y_pred),
            mae(y_true, y_pred),
            smape(y_true, y_pred),
            mape_thresh(y_true, y_pred),
        )
    )

    if RUN_FEATURE_IMPORTANCE:
        print("\n=== Permutation Feature Importance ===")
        importance_results = permutation_feature_importance(
            model=model,
            X_test=X_test,
            y_true=y_true,
            from_norm=from_norm,
            feature_names=feature_cols,
            device=DEVICE,
            repeats=3,
            seed=SEED,
        )

        for row in importance_results:
            print(
                f"{row['feature']:>22s} | "
                f"Base RMSE: {row['base_rmse']:.2f} | "
                f"Permuted RMSE: {row['permuted_rmse']:.2f} | "
                f"Increase: {row['rmse_increase']:.2f}"
            )

    plot_predictions(ts, y_true, y_pred)
    plot_daily_trend(ts, y_true, y_pred)


if __name__ == "__main__":
    main()