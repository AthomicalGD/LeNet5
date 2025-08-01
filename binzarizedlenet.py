import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets import MNIST 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

class BinaryConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0):
        binarizedInput = input.sign()
        binarizedWeights = weight.sign()  
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        out = F.conv2d(binarizedInput, binarizedWeights, bias, stride=stride, padding=padding)
        return out.sign()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]: #gradiente da entrada (a ser propagado)
            #grad_input = grad_output.mm(weight)
            grad_input = F.conv_transpose2d(grad_output, weight, bias=None, stride=stride, padding=padding)
        if ctx.needs_input_grad[1]: #gradiente de peso
            grad_weight = nn.grad.conv2d_weight(input, weight.shape, grad_output, stride=stride, padding=padding)
        if ctx.needs_input_grad[2]: #bias (soma dos gradientes)
            grad_bias = grad_output.sum(dim=[0,2,3])
        
        return grad_input, grad_weight, grad_bias, None, None
class BinaryConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.padding = padding
        self.stride = stride
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        
    def forward(self, x):
        return BinaryConvFunction.apply(x, self.weight, self.bias, self.stride, self.padding)
class BinaryFunction(torch.autograd.Function):
    def forward(ctx, input, weight, bias=None):
        binarizedInput = input.sign()
        binarizedWeights = weight.sign()
        ctx.save_for_backward(input, weight, bias)
        out = F.linear(binarizedInput, binarizedWeights, bias)
        return out
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]: #gradiente da entrada (a ser propagado)
            #grad_input = grad_output.mm(weight)
            grad_input = F.linear(grad_output, weight.t())
        if ctx.needs_input_grad[1]: #gradiente de peso
            grad_weight = grad_output.t().mm(input)
        if ctx.needs_input_grad[2]: #bias (soma dos gradientes)
            grad_bias = grad_output.sum(dim=0)
        
        return grad_input, grad_weight, grad_bias
    
class BinaryLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
    def forward(self, x):
        return BinaryFunction.apply(x, self.weight, self.bias)
class BinarizedLeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 6, 5, padding=0)
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.ModuleList([
        BinaryConv(3, 1, 5, padding=0),
        BinaryConv(3, 1, 5, padding=0),
        BinaryConv(3, 1, 5, padding=0),
        BinaryConv(3, 1, 5, padding=0),
        BinaryConv(3, 1, 5, padding=0),
        BinaryConv(3, 1, 5, padding=0),
        BinaryConv(4, 1, 5, padding=0),
        BinaryConv(4, 1, 5, padding=0),
        BinaryConv(4, 1, 5, padding=0),
        BinaryConv(4, 1, 5, padding=0),
        BinaryConv(4, 1, 5, padding=0),
        BinaryConv(4, 1, 5, padding=0),
        BinaryConv(4, 1, 5, padding=0),
        BinaryConv(4, 1, 5, padding=0),
        BinaryConv(4, 1, 5, padding=0),
        BinaryConv(6, 1, 5, padding=0)
        ])
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c5 = BinaryConv(16, 120, 5, padding=0)
        self.f6 = nn.Linear(120, 84)
        #self.f6 = BinaryLinear(120, 84, bias=True)
        #self.h = nn.ReLU()
        #self.rbf = rbf(in_features=84, num_classes=10, gamma=0.5)
        self.rbf = nn.Linear(84, 10)  # temporariamente

    
    def forward(self, x):
        x = self.s2(torch.relu(self.c1(x)))
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
        x = self.s4(x)
        #print(f"Shape após s4: {x.shape}")
        x = (self.c5(x))
        #print(f"Shape após c5: {x.shape}")
        x = torch.flatten(x, 1)
        x = (self.f6(x)) * 1.7519
        x = torch.tanh(x)
        x = self.rbf(x)
        x = torch.relu(x)
        return x
    
class BinarizedRBF(nn.Module):
    def __init__(self, in_features, num_classes, gamma=0.05):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, in_features))
        self.gamma = gamma
    
    def forward(self, x): #O unsqueeze() facilita a operação entre os vetores
        x = x.unsqueeze(1) #x = [Batch, 1,  nº de features]
        centers = self.centers.unsqueeze(0) #centers = [1, nº de classes, nº de features]
        dist = ((x - centers)**2).sum(dim=2)  #Especificar a dimensão 2 (das features)
        out = torch.exp(-self.gamma * dist)
        return out
    
class BinarizedLeNet5RBF(nn.Module):
    def __init__(self):
        super().__init__()
        self.lenet = BinarizedLeNet5()
        self.rbf = BinarizedRBF(in_features=84, num_classes=10, gamma=0.05)
    
    def forward(self, x):
        x = self.lenet(x)
        #x = self.rbf(x)
        #print(f"Formato após total: {x.shape}") #[batch_size, num_classes]
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
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False)

    print(f"Carregou os dados")

    model = BinarizedLeNet5RBF().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    print(f"Inicializou o modelo e vai entrar no treinamento")

    for i in range(epochs):
        model.train()
        for batch, (data, target) in enumerate(training_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.to(device), target)
            loss.backward()
            optimizer.step()
            print(f"Erro médio: {loss} - Época {i}")
    model.eval()
    data = iter(test_loader)
    images, labels = next(data)
    images = images.to(device)
    labels = labels.to(device)
    samples = 100
    erros = 0
    with torch.no_grad():
        for i in range(samples):
            img = images[i].unsqueeze(0)  # adiciona dimensão de batch (1, C, H, W)
            label = labels[i].item()

            # Faz a inferência
            output = model(img)
            prediction = output.argmax(dim=1).item()
            print(f"Valor: {label} Predição: {prediction}")
            if label != prediction: erros += 1
    print(f"Erros: {erros}")