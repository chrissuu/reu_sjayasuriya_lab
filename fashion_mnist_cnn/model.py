import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import torch.optim as optim
import torch.nn as nn

from cnn import ConvNet
from train import Trainer
from test import Tester

training_data = datasets.FashionMNIST(
        root = "data",
        train = True,
        download = True,
        transform = ToTensor()
)

test_data = datasets.FashionMNIST(
        root = "data",
        train = False,
        download = True,
        transform = ToTensor()
)

train_loader = DataLoader(training_data, batch_size = 64, shuffle = True)
test_loader = DataLoader(test_data, batch_size = 64, shuffle = True)
print("training data ready!\n")

device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
)

print("device: " + str(device) + "\n")

net = ConvNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr= 0.001)

trainer = Trainer(3, train_loader, criterion, optimizer)
trainer.train(net)

print("training done!")

tester = Tester(test_loader, net)
tester.test()

