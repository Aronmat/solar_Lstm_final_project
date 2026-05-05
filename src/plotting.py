import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_training_loss(losses):
    plt.figure()
    plt.plot(losses)
    plt.title("Solar — Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_loss.png", dpi=150)
    plt.show()


def plot_predictions(ts, y_true, y_pred):
    n_thin = 12

    ts_th = ts[::n_thin]
    y_t = y_true[::n_thin]
    y_p = y_pred[::n_thin]

    plt.figure(figsize=(14, 6))
    plt.plot(ts_th, y_t, label="Solar Actual", linewidth=1.5)
    plt.plot(ts_th, y_p, label="Solar Predicted", linewidth=1.5, linestyle="--")
    plt.title("Solar Generation — Test Set")
    plt.xlabel("Date")
    plt.ylabel("Power")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    plt.savefig("solar_plot_test_set.png", dpi=150)
    plt.show()


def plot_daily_trend(ts, y_true, y_pred):
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
    plt.title("Solar Generation — Daily Trend")
    plt.xlabel("Date")
    plt.ylabel("Power")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, frameon=False)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    plt.savefig("solar_plot_daily_trend.png", dpi=150)
    plt.show()