import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset


class LNRBF(nn.Module):
    def __init__(self, in_features, num_classes, gamma=1.0):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, in_features,))
        self.gamma = gamma
    
    def forward(self, x): #O unsqueeze() facilita a operação entre os vetores
        x = x.unsqueeze(1) #x = [Batch, 1,  nº de features]
        centers = self.centers.unsqueeze(0) #centers = [1, nº de classes, nº de features]
        dist = ((x - centers)**2).sum(dim=2)  #Especificar a dimensão 2 (das features)
        out = torch.exp(-self.gamma * dist)
        return out

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, 5, padding=0)
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = [None] * 16
        self.c3[0] = nn.Conv2d(3, 1, 5, padding=0)
        self.c3[1] = nn.Conv2d(3, 1, 5, padding=0)
        self.c3[2] = nn.Conv2d(3, 1, 5, padding=0)
        self.c3[3] = nn.Conv2d(3, 1, 5, padding=0)
        self.c3[4] = nn.Conv2d(3, 1, 5, padding=0)
        self.c3[5] = nn.Conv2d(3, 1, 5, padding=0)
        self.c3[6] = nn.Conv2d(4, 1, 5, padding=0)
        self.c3[7] = nn.Conv2d(4, 1, 5, padding=0)
        self.c3[8] = nn.Conv2d(4, 1, 5, padding=0)
        self.c3[9] = nn.Conv2d(4, 1, 5, padding=0)
        self.c3[10] = nn.Conv2d(4, 1, 5, padding=0)
        self.c3[11] = nn.Conv2d(4, 1, 5, padding=0)
        self.c3[12] = nn.Conv2d(4, 1, 5, padding=0)
        self.c3[13] = nn.Conv2d(4, 1, 5, padding=0)
        self.c3[14] = nn.Conv2d(4, 1, 5, padding=0)
        self.c3[15] = nn.Conv2d(6, 1, 5, padding=0)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c5 = nn.Conv2d(16, 120, 5, padding=0)
        self.f6 = nn.Linear(120, 84)
        self.rbf = LNRBF(in_features=84, num_classes=10, gamma=0.5)
    
    def forward(self, x):
        x = self.s2(torch.relu(self.c1(x)))
        x = torch.sigmoid(x)
        #print(f"Shape após s2: {x.shape}")
        # x: shape [batch, 6, H, W]
        fmapc3 = [None] * 16
        fmapc3[0] = x[:, [0, 1, 2], :, :]
        fmapc3[1] = x[:, [1, 2, 3], :, :]
        fmapc3[2] = x[:, [2, 3, 4], :, :]
        fmapc3[3] = x[:, [3, 4, 5], :, :]
        fmapc3[4] = x[:, [0, 4, 5], :, :]
        fmapc3[5] = x[:, [0, 1, 5], :, :]
        fmapc3[6] = x[:, [0, 1, 2, 3], :, :]
        fmapc3[7] = x[:, [1, 2, 3, 4], :, :]
        fmapc3[8] = x[:, [2, 3, 4, 5], :, :]
        fmapc3[9] = x[:, [0, 3, 4, 5], :, :]
        fmapc3[10] = x[:, [0, 1, 4, 5], :, :]
        fmapc3[11] = x[:, [0, 1, 2, 5], :, :]
        fmapc3[12] = x[:, [0, 1, 3, 4], :, :]
        fmapc3[13] = x[:, [1, 2, 4, 5], :, :]
        fmapc3[14] = x[:, [0, 2, 3, 5], :, :]
        fmapc3[15] = x[:, [0, 1, 2, 3, 4, 5], :, :]
        out = [None] * 16
        for i in range(16):
            out[i] = self.c3[i](fmapc3[i])
        x = torch.cat([out[i] for i in range(16)], dim=1)
        x = torch.relu(x)
        #print(f"Shape após c3: {x.shape}")
        x = torch.sigmoid(self.s4(x))
        #print(f"Shape após s4: {x.shape}")
        x = torch.relu(self.c5(x))
        #print(f"Shape após c5: {x.shape}")
        x = torch.flatten(x, 1)
        x = torch.tanh(self.f6(x))
        x = self.rbf(x)
        return x

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #*Transformação de imagem 28x28 para tensor 32x32 (normalizado com média e desvio padrão do próprio mnist)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])

    print(f"Vai carregar os dados (passou do módulo)")

    epochs = 1

    training_set = MNIST(root='/.data', train=True, download=True, transform=transform)
    test_set =  MNIST(root='/.data', train=False, download=True, transform=transform)

    training_loader = DataLoader(training_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(training_set, batch_size=10, shuffle=False)

    print(f"Carregou os dados")

    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print(f"Inicializou o modelo e vai entrar no treinamento")

    for i in range(epochs):
        model.train()
        for batch, (data, target) in enumerate(training_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    print(f"Época {epochs + 1}: Erro médio: {loss.item()}")

    x = enumerate(test_loader)
    for i in range(epochs):
        model.eval()
        for batch, (data, target) in x:
            data, target = data.to(device), target.to(device)
            output = model(data)
            print(f"Saída esperada: {target} - Saída: {output}")