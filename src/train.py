import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

from src.config import (
    TRAIN_PATH,
    PRED_PATH,
    TIMESTEPS,
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
    TARGET,
    USE_SPLINE,
)

from src.data_preprocessing import (
    load_pair,
    make_normalizers,
    make_sequences_with_target,
    time_ordered_split,
)

from src.lstm_model import LSTMReg
from src.evaluate import rmse, mae, smape, mape_thresh
from src.plotting import plot_training_loss, plot_predictions, plot_daily_trend


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_df, pred_df, pc_cols = load_pair(
        TRAIN_PATH,
        PRED_PATH,
        TARGET,
        use_spline=USE_SPLINE,
    )

    print(f"[Solar LSTM] Using target: {TARGET}")
    print(f"[Solar LSTM] PCs: {pc_cols}")
    print(f"[Solar LSTM] Cubic spline enabled: {USE_SPLINE}")

    to_norm, from_norm = make_normalizers(train_df, TARGET)

    X_all, Y_all = make_sequences_with_target(
        train_df,
        pc_cols,
        TARGET,
        TIMESTEPS,
        to_norm,
    )

    X_train, Y_train, X_val, Y_val, _, _ = time_ordered_split(X_all, Y_all)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(Y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    step_times = pd.to_datetime(pred_df["timestamp"].values, utc=True)
    for i in range(min(3, len(pred_df) - TIMESTEPS - 1)):
        assert step_times[i + TIMESTEPS] > step_times[i + TIMESTEPS - 1], (
            "Leakage check failed: target time is not after history end!"
        )

    print("[Solar LSTM] Alignment check: OK")

    input_size = len(pc_cols) + 1
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

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss += float(loss.item())

        model.eval()

        with torch.no_grad():
            if len(X_val):
                vloss = criterion(
                    model(torch.tensor(X_val).to(DEVICE)),
                    torch.tensor(Y_val).to(DEVICE),
                ).item()
            else:
                vloss = epoch_loss

        sched.step(vloss)

        if epoch % 10 == 0:
            print(
                f"[Solar LSTM] Epoch {epoch:03d} | "
                f"TrainLoss {epoch_loss:.2f} | ValLoss {vloss:.4f}"
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

    Xp, _ = make_sequences_with_target(
        pred_df,
        pc_cols,
        TARGET,
        TIMESTEPS,
        to_norm,
    )

    model.eval()
    with torch.no_grad():
        yhat_norm = model(torch.tensor(Xp).to(DEVICE)).cpu().numpy().flatten()

    y_pred = from_norm(yhat_norm)
    y_true = pred_df[TARGET].values[TIMESTEPS:TIMESTEPS + len(y_pred)]
    y_pred = np.clip(y_pred, 0.0, None)

    ts = pd.to_datetime(pred_df["timestamp"], utc=True).iloc[
        TIMESTEPS:TIMESTEPS + len(y_pred)
    ]

    mask = np.isfinite(y_true) & np.isfinite(y_pred) & pd.Series(ts).notna().values
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    ts = ts.iloc[mask.nonzero()[0]]

    T_pred = pred_df[TARGET].values.astype(float)
    naive = []
    true_baseline = []

    for i in range(len(pred_df) - TIMESTEPS):
        true_baseline.append(T_pred[i + TIMESTEPS])
        naive.append(T_pred[i + TIMESTEPS - 1])

    true_baseline = np.asarray(true_baseline)
    naive = np.asarray(naive)

    valid = np.isfinite(true_baseline) & np.isfinite(naive)
    true_baseline = true_baseline[valid]
    naive = naive[valid]

    print("\n=== Baseline vs Model ===")
    print(
        "Naive  -> RMSE: %.2f | MAE: %.2f | SMAPE: %.2f%%"
        % (rmse(true_baseline, naive), mae(true_baseline, naive), smape(true_baseline, naive))
    )

    print(
        "Model  -> RMSE: %.2f | MAE: %.2f | SMAPE: %.2f%% | MAPE@>=10MW: %.2f%%"
        % (rmse(y_true, y_pred), mae(y_true, y_pred), smape(y_true, y_pred), mape_thresh(y_true, y_pred))
    )

    plot_predictions(ts, y_true, y_pred)
    plot_daily_trend(ts, y_true, y_pred)


if __name__ == "__main__":
    main()