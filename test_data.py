
from torch.utils.data import TensorDataset
from model import SASRec
from train import train_model
import torch

seq = torch.tensor([
    [0, 0, 1, 2, 3],
    [0, 4, 5, 6, 7]
])

pos = torch.tensor([
    [0, 0, 2, 3, 8],
    [0, 5, 6, 7, 9]
])

neg = torch.tensor([
    [0, 0, 10, 11, 12],
    [0, 13, 14, 15, 16]
])

dataset = TensorDataset(seq, pos, neg)

model = SASRec(
    itemnum=100,
    hidden_units=64,
    maxlen=5,
    num_heads=2,
    num_blocks=2,
    dropout_rate=0.2
)

train_model(
    model=model,
    train_dataset=dataset,
    epochs=5,
    batch_size=2,
    learning_rate=0.001
)
