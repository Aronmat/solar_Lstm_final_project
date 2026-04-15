import torch
from src.lstm_model import LSTMRegressor


def test_model_output_shape():
    model = LSTMRegressor(input_size=5)
    x = torch.randn(8, 10, 5)
    y = model(x)

    assert y.shape == (8, 1)