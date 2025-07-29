import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import datasets
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

base = datasets.load_breast_cancer()

x = torch.Tensor(base.data)
y = torch.Tensor(base.target).unsqueeze(1)
dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=32) #SHUFFLE DESLIGADO

model = nn.Sequential(
    nn.Linear(30, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)

epochs = 1000

for i in range(epochs):
    for entry, target in loader:
        output = model(entry)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()