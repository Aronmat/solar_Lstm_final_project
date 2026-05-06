from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

HISTORICAL_DATA_PATH = ROOT / "data" / "solar_energy_data.csv.xlsx"
LOCAL_DATA_DIR = ROOT / "data" / "local"
LOCAL_FILE_PATTERN = "gpg_*.csv"

HISTORICAL_TARGET = "DE_solar_generation_actual"
LOCAL_TARGET = "PV_Power_W"

HISTORICAL_FEATURE_COLS = [
    "SWTDN",
    "SWGDN",
    "cloudcover",
    "humidity",
    "dew",
]

LOCAL_FEATURE_COLS = [
    "Solar_Irradiance_Wm2",
    "PV_Current_A",
    "PV_Voltage_V",
    "Battery_Power_W",
    "Avg_Power_to_Battery_W",
]
HISTORICAL_TIMESTEPS = 72
HISTORICAL_FORECAST_HORIZON = 1

LOCAL_TIMESTEPS = 60
LOCAL_FORECAST_HORIZON = 5

HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.15
EPOCHS = 120
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 12
SEED = 42
DEVICE = "cpu"

USE_SPLINE = True
SHOW_BATCH_PROGRESS = True
BATCH_PRINT_EVERY = 10
RUN_FEATURE_IMPORTANCE = True