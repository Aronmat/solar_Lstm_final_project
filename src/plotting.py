import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_training_loss(losses):
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_loss.png", dpi=150)
    plt.show()


def plot_predictions(ts, y_true, y_pred, mode="historical"):
    if mode == "local":
        plot_local_predictions(ts, y_true, y_pred)
    else:
        plot_historical_predictions(ts, y_true, y_pred)


def plot_historical_predictions(ts, y_true, y_pred):
    n_thin = 12

    ts_th = ts[::n_thin]
    y_t = y_true[::n_thin]
    y_p = y_pred[::n_thin]

    plt.figure(figsize=(14, 6))
    plt.plot(ts_th, y_t, label="Solar Actual", linewidth=1.5)
    plt.plot(ts_th, y_p, label="Solar Predicted", linewidth=1.5, linestyle="--")
    plt.title("Historical Solar Generation — Test Set")
    plt.xlabel("Date")
    plt.ylabel("Power")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    plt.savefig("historical_solar_predictions.png", dpi=150)
    plt.show()


def plot_local_predictions(ts, y_true, y_pred):
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(ts),
        "actual": y_true,
        "predicted": y_pred,
    }).dropna().sort_values("timestamp")

    df["error"] = df["actual"] - df["predicted"]

    # Plot 1: Full local prediction window
    plt.figure(figsize=(14, 6))
    plt.plot(df["timestamp"], df["actual"], label="Actual PV Power", linewidth=1.5)
    plt.plot(df["timestamp"], df["predicted"], label="Predicted PV Power", linewidth=1.5, linestyle="--")
    plt.title("Local Solar Power Prediction — Full Test Window")
    plt.xlabel("Time")
    plt.ylabel("Power (W)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig("local_predictions_full.png", dpi=150)
    plt.show()

    # Daylight-only filter so nighttime zeros do not dominate the plot
    daylight = df[df["actual"] > 10].copy()

    if len(daylight) > 0:
        plt.figure(figsize=(14, 6))
        plt.plot(daylight["timestamp"], daylight["actual"], label="Actual PV Power", linewidth=1.5)
        plt.plot(daylight["timestamp"], daylight["predicted"], label="Predicted PV Power", linewidth=1.5, linestyle="--")
        plt.title("Local Solar Power Prediction — Daylight Only")
        plt.xlabel("Time")
        plt.ylabel("Power (W)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        plt.savefig("local_predictions_daylight_only.png", dpi=150)
        plt.show()

    # Plot 3: Error over time
    plt.figure(figsize=(14, 5))
    plt.plot(df["timestamp"], df["error"], label="Prediction Error", linewidth=1.2)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title("Local Solar Prediction Error Over Time")
    plt.xlabel("Time")
    plt.ylabel("Actual - Predicted (W)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig("local_prediction_error.png", dpi=150)
    plt.show()

    # Plot 4: Actual vs predicted scatter
    plt.figure(figsize=(7, 7))
    plt.scatter(df["actual"], df["predicted"], alpha=0.5)

    min_val = min(df["actual"].min(), df["predicted"].min())
    max_val = max(df["actual"].max(), df["predicted"].max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1)

    plt.title("Local Solar Prediction — Actual vs Predicted")
    plt.xlabel("Actual PV Power (W)")
    plt.ylabel("Predicted PV Power (W)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("local_actual_vs_predicted.png", dpi=150)
    plt.show()


def plot_daily_trend(ts, y_true, y_pred, mode="historical"):
    if mode == "local":
        plot_local_rolling_trend(ts, y_true, y_pred)
    else:
        plot_historical_daily_trend(ts, y_true, y_pred)


def plot_historical_daily_trend(ts, y_true, y_pred):
    df_plot = pd.DataFrame({
        "ts": pd.to_datetime(ts, utc=True),
        "actual": y_true,
        "pred": y_pred,
    }).dropna().sort_values("ts")

    daily = df_plot.set_index("ts").resample("D").mean()
    daily["actual7"] = daily["actual"].rolling(7, center=True, min_periods=1).mean()
    daily["pred7"] = daily["pred"].rolling(7, center=True, min_periods=1).mean()

    plt.figure(figsize=(14, 6))
    plt.plot(daily.index, daily["actual7"], label="Solar Actual (7-day mean)", linewidth=2)
    plt.plot(daily.index, daily["pred7"], label="Solar Predicted (7-day mean)", linewidth=2, linestyle="--")
    plt.title("Historical Solar Generation — Daily Trend")
    plt.xlabel("Date")
    plt.ylabel("Power")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, frameon=False)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    plt.savefig("historical_daily_trend.png", dpi=150)
    plt.show()


def plot_local_rolling_trend(ts, y_true, y_pred):
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(ts),
        "actual": y_true,
        "predicted": y_pred,
    }).dropna().sort_values("timestamp")

    df = df.set_index("timestamp")

    # 15-minute rolling average is more useful than daily average for local minute data
    rolling = df.rolling("15min", min_periods=1).mean()

    plt.figure(figsize=(14, 6))
    plt.plot(rolling.index, rolling["actual"], label="Actual PV Power (15-min mean)", linewidth=2)
    plt.plot(rolling.index, rolling["predicted"], label="Predicted PV Power (15-min mean)", linewidth=2, linestyle="--")
    plt.title("Local Solar Power Prediction — 15-Minute Rolling Trend")
    plt.xlabel("Time")
    plt.ylabel("Power (W)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig("local_15min_rolling_trend.png", dpi=150)
    plt.show()