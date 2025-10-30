import numpy as np
import cv2 as cv
import ultralytics as ul
import pandas as pd
import matplotlib as mt #Statistical chart of how often the cars are present in the footage
import torch
from torch import save, load
import torchvision as tv
import requests
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from HondaDataset import Honda
from torch.utils.data import DataLoader
from torch.optim import Adam



transform = transforms.Compose([
    transforms.Resize((480, 640)),  # or (224, 224) for smaller models
    transforms.ToTensor()
])

DataHonda = Honda(
    csv_file='/home/waqar/Programming/CarView/honda_image_dataset_v2/HondaCarsV1_cleaned.csv',
    root_dir='/home/waqar/Programming/CarView/Honda_resized',
    transform=transform
)
#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 32
#Load custom Honda Datasets
image_size = 10354


DataHonda = Honda(csv_file = '/home/waqar/Programming/CarView/honda_image_dataset_v2/HondaCarsV1_cleaned.csv', root_dir = '/home/waqar/Programming/CarView/Honda_resized', transform=transform)

train_size = int(0.8 * len(DataHonda))
test_size = len(DataHonda) - train_size
train_honda_set, test_honda_set = torch.utils.data.random_split(DataHonda, [train_size, test_size])
honda_train_loader = DataLoader(dataset=train_honda_set, batch_size=batch_size,shuffle=True)
honda_test_loader = DataLoader(dataset=test_honda_set, batch_size=batch_size,shuffle=False)

img, label = DataHonda[0]
print(img.shape)

classes = (
    'Accord',
    'City',
    'Civic',
    'Pilot',
    'Passport',
    'NSX',
    'Odyssey'
) #7 classes

class HondaClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),  # -> 240x320
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),  # -> 120x160
                nn.Flatten(),
                nn.Linear(64 * 120 * 160, 128),  # smaller dense layer
                nn.ReLU(),
                nn.Linear(128, 7)  # 7 classes
            )

        def forward(self, x):
            return self.model(x)

# instance of neural network, loss, optimizer
hondaclf = HondaClassifier().to(device)
hondaopt = Adam(hondaclf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

#Training
if __name__ == "__main__":
    for epoch in range(10): #Go through the batch 10 times
        for batch in honda_train_loader:
            X,y = batch
            X,y = X.to(device),y.to(device)
            yhat= hondaclf(X)
            loss = loss_fn(yhat, y)

            #backpropagation
            hondaopt.zero_grad()
            loss.backward()
            hondaopt.step()

        print(f"Epoch {epoch} loss is {loss.item()}")

    with open('model_state.pt', 'wb') as f:
        save(hondaclf.state_dict(), f)






