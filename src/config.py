from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT / "data" / "pca_trainer_with_solar_HALFYEAR.csv"
PRED_PATH  = ROOT / "data" / "pca_predictor_with_solar_HALFYEAR.csv"

TARGET = "solar_generation_mw"

TIMESTEPS = 72
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