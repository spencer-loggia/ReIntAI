import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import torch.nn.functional as F

train_data = datasets.MNIST(
    root='./data', train=True, transform=ToTensor(), download=True
)
test_data = datasets.MNIST(
    root='./data', train=False, transform=ToTensor(), download=True
)

loaders = {
    'train': torch.utils.data.DataLoader(train_data,
                                         batch_size=100,
                                         shuffle=True,
                                         num_workers=1),

    'test': torch.utils.data.DataLoader(test_data,
                                        batch_size=100,
                                        shuffle=True,
                                        num_workers=1),
}

sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

class RNN_MNIST(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_MNIST, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set Initialized State
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Passing in the input and hidden state into the model and  obtaining outputs
        out, hidden = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = self.fc(out[:, -1, :])
        return out

    def train(number_of_epochs, ):
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)



model = RNN_MNIST(input_size, hidden_size, num_layers, num_classes).to(device).to(device)

