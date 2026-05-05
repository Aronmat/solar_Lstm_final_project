import torch
from src.lstm_model import LSTMReg


def test_model_output_shape():
    model = LSTMReg(input_size=5, hidden_size=64, num_layers=2, dropout=0.1)
    x = torch.randn(8, 10, 5)

    y = model(x)

    assert y.shape == (8, 1)