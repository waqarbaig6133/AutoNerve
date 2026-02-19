import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision import models
from google.colab import drive
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
drive.mount('/content/drive')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# -------------------------------------------------------
# TRANSFORMS (ImageNet normalization + lighter augment)
# -------------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------------------------------
# DATASET + VALIDATION SPLIT
# -------------------------------------------------------
full_dataset = torchvision.datasets.ImageFolder(
    root='/content/drive/MyDrive/honda_cars',
    transform=train_transform
)

val_size = int(len(full_dataset) * 0.1)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# validation must NOT have augmentations
val_dataset.dataset.transform = test_transform

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

class_names = full_dataset.classes
num_classes = len(class_names)


# -------------------------------------------------------
# MODEL (ResNet18)
# -------------------------------------------------------
net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_features = net.fc.in_features
net.fc = nn.Linear(num_features, num_classes)
net = net.to(device)


# -------------------------------------------------------
# STAGE 1 — TRAIN FC ONLY
# -------------------------------------------------------
for param in net.parameters():
    param.requires_grad = False
for param in net.fc.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.fc.parameters(), lr=1e-3)

print("Stage 1: training classifier head only...")
for epoch in range(5):
    net.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/5 | Loss: {running_loss/len(train_loader):.4f}")


# -------------------------------------------------------
# STAGE 2 — UNFREEZE EVERYTHING + lower LR
# -------------------------------------------------------
for param in net.parameters():
    param.requires_grad = True

optimizer = optim.Adam(net.parameters(), lr=1e-4)

print("Stage 2: fine-tuning entire model...")
epochs = 25  # total 30

for epoch in range(epochs):
    net.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"[Epoch {epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f}")


print("Training complete.")


# -------------------------------------------------------
# PREDICTION FUNCTION
# -------------------------------------------------------
def predict_image(image_path, model, device, top_k=3):
    model.eval()

    transform = test_transform
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)

    top_probs, top_indices = probs.topk(top_k, dim=1)
    top_probs = top_probs.cpu().numpy().flatten()
    top_indices = top_indices.cpu().numpy().flatten()

    top_class = class_names[top_indices[0]]
    confidence = top_probs[0]
    other_guesses = [(class_names[i], p) for i, p in zip(top_indices[1:], top_probs[1:])]

    return top_class, confidence, other_guesses
