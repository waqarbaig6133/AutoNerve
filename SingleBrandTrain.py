import torch
import torch.nn as nn
import torch.nn.modules
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, dataloader
import PIL
from PIL import Image
import os
from PIL import UnidentifiedImageError

from google.colab import drive
drive.mount('/content/drive')

device = torch.device("cpu")
if torch.cuda.is_available():
  device = torch.device("cuda")
print(device)

transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
]) # Preprocess images. Tensor has value from 0 to 1, the normalize from -1 to 1, instead of so many colors

train_dataset = torchvision.datasets.ImageFolder(root = '/content/drive/MyDrive/honda_cars', transform = transform)
# Filter out dummy samples with label -1
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)

class HondaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d((6, 6))  # makes it work for any input size
        self.fc1 = nn.Linear(128 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 17)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

net = HondaNet()
net.to(device)

loss_function = nn.NLLLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    running_loss = 0.0

    for x, y in enumerate(train_loader):
        inputs, labels = y[0].to(device), y[1].to(device)


        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if x % 2000 == 1999:  # print every 2000 batches
            print(f"[Epoch {epoch+1}/{epochs}, Batch {x+1}] loss: {running_loss / 2000:.6f}")
            running_loss = 0.0

print("Finished Training")
