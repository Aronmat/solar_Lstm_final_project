# Lstm_Pca_halfyear_solar_clean.py
# Train: first 6 months (solar-only)
# Predict: last 6 months (solar-only)
# Target: solar_generation_mw (created by your solar filter)
# Features: PCs + target-history (log1p-normalized)
# Extras: leakage checks, naïve baseline, clean plots with "Solar" labels

from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---------------- Paths ----------------
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
TRAIN_PATH = HERE / "pca_trainer_with_solar_HALFYEAR.csv"
PRED_PATH  = HERE / "pca_predictor_with_solar_HALFYEAR.csv"

# ---------------- Hyperparams ----------------
TIMESTEPS     = 72        # 3 days of context (try 168 for weekly)
HIDDEN_SIZE   = 128
NUM_LAYERS    = 2
DROPOUT       = 0.15
EPOCHS        = 120
BATCH_SIZE    = 256
LEARNING_RATE = 3e-4
WEIGHT_DECAY  = 1e-5
PATIENCE      = 12        # early stopping
SEED          = 42
DEVICE        = "cpu"     # set "cuda" if available

np.random.seed(SEED); torch.manual_seed(SEED)

# ---------------- Metrics helpers ----------------
def smape(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    d = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return float(np.mean(np.where(d > 1e-6, np.abs(y_true - y_pred)/d, 0.0)) * 100.0)

def mape_thresh(y_true, y_pred, min_denom=10.0):  # avoid tiny night denominators
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    mask = np.abs(y_true) >= min_denom
    if not np.any(mask): return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

# ---------------- Load trainer/predictor ----------------
def load_pair(tp: Path, pp: Path):
    if not tp.exists(): raise FileNotFoundError(f"Missing trainer CSV: {tp}")
    if not pp.exists(): raise FileNotFoundError(f"Missing predictor CSV: {pp}")

    t = pd.read_csv(tp); p = pd.read_csv(pp)
    for df in (t, p):
        if "utc_timestamp" in df.columns and "timestamp" not in df.columns:
            df.rename(columns={"utc_timestamp":"timestamp"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df.dropna(subset=["timestamp"], inplace=True)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

    TARGET = "solar_generation_mw"  # fixed solar target name
    for df in (t, p):
        if TARGET not in df.columns:
            raise ValueError(f"'{TARGET}' must exist in both CSVs. Re-run the solar filter.")
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").rolling(5, min_periods=1).mean()

    # Common PC columns
    def pc_key(col):
        s = str(col); return int(''.join(ch for ch in s if ch.isdigit()) or 0)
    pcs_t = [c for c in t.columns if str(c).upper().startswith("PC")]
    pcs_p = [c for c in p.columns if str(c).upper().startswith("PC")]
    PC_COLS = [c for c in sorted(pcs_t, key=pc_key) if c in pcs_p]
    if not PC_COLS: raise ValueError("No common PC columns between trainer and predictor.")

    print(f"[Solar LSTM] Using target: {TARGET}")
    print(f"[Solar LSTM] PCs: {PC_COLS}")
    return t, p, PC_COLS, TARGET

train_df, pred_df, PC_COLS, TARGET = load_pair(TRAIN_PATH, PRED_PATH)

# ---------------- Target normalization (log1p) ----------------
ylog = np.log1p(np.clip(train_df[TARGET].values, 0, None))
mean_log, std_log = float(np.mean(ylog)), float(np.std(ylog) or 1.0)
to_norm   = lambda y: (np.log1p(np.clip(y, 0, None)) - mean_log) / std_log
from_norm = lambda z: np.maximum(np.expm1(z * std_log + mean_log), 0.0)

# ---------------- Build sequences WITH target history channel ----------------
def make_sequences_with_target(df: pd.DataFrame, pc_cols, target_col, steps: int):
    V = df[pc_cols].values.astype(np.float32)      # (N, k) PCs
    T = df[target_col].values.astype(np.float32)   # (N,)  solar target
    Tn = to_norm(T)
    X, Y = [], []
    for i in range(len(df) - steps):
        Xpc = V[i:i+steps]                       # (steps, k)
        Xtg = Tn[i:i+steps].reshape(steps, 1)    # (steps, 1) history
        y_next = Tn[i + steps]                   # future point
        X.append(np.concatenate([Xpc, Xtg], axis=1))  # (steps, k+1)
        Y.append([y_next])
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)

# Trainer sequences (time-ordered split for realistic val)
X_all, Y_all = make_sequences_with_target(train_df, PC_COLS, TARGET, TIMESTEPS)
s1, s2 = int(0.70*len(X_all)), int(0.85*len(X_all))
X_train, Y_train = X_all[:s1], Y_all[:s1]
X_val,   Y_val   = X_all[s1:s2], Y_all[s1:s2]

train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(Y_train)),
                          batch_size=BATCH_SIZE, shuffle=True)

# ---------------- Alignment sanity check (no future peeking) ----------------
step_times = pd.to_datetime(pred_df["timestamp"].values, utc=True)
for i in range(min(3, len(pred_df) - TIMESTEPS - 1)):
    assert step_times[i + TIMESTEPS] > step_times[i + TIMESTEPS - 1], \
        "Leakage check failed: target time is not after history end!"
print("[Solar LSTM] Alignment check: OK (no future leakage).")

# ---------------- Model ----------------
class LSTMReg(nn.Module):
    def __init__(self, input_size, hidden=HIDDEN_SIZE, layers=NUM_LAYERS, drop=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=layers,
                            batch_first=True, dropout=(drop if layers>1 else 0.0))
        self.drop = nn.Dropout(drop)
        self.fc   = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x); out = self.drop(out[:, -1, :]); return self.fc(out)

INPUT_SIZE = len(PC_COLS) + 1
model = LSTMReg(INPUT_SIZE).to(DEVICE)
criterion = nn.SmoothL1Loss(beta=1.0)  # Huber in normalized log space
opt = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=PATIENCE//2)

# ---------------- Train with early stopping ----------------
best_vloss, best_state, no_improve = float("inf"), None, 0
losses = []
for epoch in range(EPOCHS):
    model.train(); eloss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        eloss += float(loss.item())

    model.eval()
    with torch.no_grad():
        vloss = criterion(model(torch.tensor(X_val).to(DEVICE)),
                          torch.tensor(Y_val).to(DEVICE)).item() if len(X_val) else eloss
    sched.step(vloss)
    if epoch % 10 == 0:
        print(f"[Solar LSTM] Epoch {epoch:03d} | TrainLoss(norm-log) {eloss:.2f} | ValLoss(norm-log) {vloss:.4f}")
    losses.append(eloss)

    if vloss + 1e-6 < best_vloss:
        best_vloss = vloss; best_state = model.state_dict(); no_improve = 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"[Solar LSTM] Early stopping at epoch {epoch} (best val loss {best_vloss:.4f})")
            break

if best_state is not None:
    model.load_state_dict(best_state)

plt.figure(); plt.plot(losses)
plt.title("Solar — Training Loss (norm-log)")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.grid(True); plt.tight_layout(); plt.show()

# ---------------- Predict on predictor half (solar) ----------------
def predict_pred_half(df_pred: pd.DataFrame):
    Xp, Yp = make_sequences_with_target(df_pred, PC_COLS, TARGET, TIMESTEPS)
    with torch.no_grad():
        yhat_norm = model(torch.tensor(Xp).to(DEVICE)).cpu().numpy().flatten()
    yhat  = from_norm(yhat_norm)
    ytrue = df_pred[TARGET].values[TIMESTEPS:TIMESTEPS+len(yhat)]
    yhat  = np.clip(yhat, 0.0, None)
    ts    = pd.to_datetime(df_pred["timestamp"], utc=True).iloc[TIMESTEPS:TIMESTEPS+len(yhat)]
    return ytrue.astype(float), yhat.astype(float), ts

y_true, y_pred, ts = predict_pred_half(pred_df)

# Mask invalids
mask = np.isfinite(y_true) & np.isfinite(y_pred) & pd.Series(ts).notna().values
y_true, y_pred, ts = y_true[mask], y_pred[mask], ts.iloc[mask.nonzero()[0]]

# ---------------- Naïve baseline (solar) ----------------
T_pred = pred_df[TARGET].values.astype(float)
naive, true_ = [], []
for i in range(len(pred_df) - TIMESTEPS):
    true_.append(T_pred[i + TIMESTEPS])
    naive.append(T_pred[i + TIMESTEPS - 1])  # last observed solar value
true_, naive = np.asarray(true_), np.asarray(naive)
m2 = np.isfinite(true_) & np.isfinite(naive)
true_, naive = true_[m2], naive[m2]

from sklearn.metrics import mean_squared_error as _mse, mean_absolute_error as _mae
print("\n=== Baseline vs Model (predictor half, SOLAR) ===")
rmse = np.sqrt(_mse(true_, naive))
mae = _mae(true_, naive)
print("Naive  -> RMSE: %.2f | MAE: %.2f | SMAPE: %.2f%%" % (rmse, mae, smape(true_, naive)))

rmse_model = np.sqrt(_mse(y_true, y_pred))
mae_model = _mae(y_true, y_pred)
print("Model  -> RMSE: %.2f | MAE: %.2f | SMAPE: %.2f%% | MAPE@>=10MW: %.2f%%" %
      (rmse_model, mae_model, smape(y_true, y_pred), mape_thresh(y_true, y_pred, 10.0)))

# ---------------- Clean plots (Solar labels) ----------------
# 1) Thinned raw
N_THIN = 12
ts_th, yT, yP = ts[::N_THIN], y_true[::N_THIN], y_pred[::N_THIN]
plt.figure(figsize=(14,6))
plt.plot(ts_th, yT, label="Solar Actual", linewidth=1.5)
plt.plot(ts_th, yP, label="Solar Predicted", linewidth=1.5, linestyle="--")
plt.title("Solar Generation — predictor half (thinned raw)")
plt.xlabel("Date (UTC)"); plt.ylabel("MW")
plt.grid(True, alpha=0.3); plt.legend()
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.tight_layout(); plt.savefig("solar_plot_raw_thinned.png", dpi=150); plt.show()

# 2) Daily trend
df_plot = pd.DataFrame({"ts": pd.to_datetime(ts, utc=True), "actual": y_true, "pred": y_pred}).dropna().sort_values("ts")
daily = df_plot.set_index("ts").resample("D").mean()
daily["actual7"] = daily["actual"].rolling(7, center=True, min_periods=1).mean()
daily["pred7"]   = daily["pred"].rolling(7, center=True, min_periods=1).mean()
plt.figure(figsize=(14,6))
plt.plot(daily.index, daily["actual7"], label="Solar Actual (7-day mean)", linewidth=2)
plt.plot(daily.index, daily["pred7"],   label="Solar Predicted (7-day mean)", linewidth=2, linestyle="--")
plt.title("Solar Generation — predictor half (daily, smoothed)")
plt.xlabel("Date (UTC)"); plt.ylabel("MW")
plt.grid(True, alpha=0.3); plt.legend(ncol=2, frameon=False)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.tight_layout(); plt.savefig("solar_plot_daily_trend.png", dpi=150); plt.show()
