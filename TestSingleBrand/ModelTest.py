import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# 1. Match the classes exactly to your training folder structure
classes = ('accord', 'amaze', 'brio', 'city', 'civic', 'clarity', 'freed',
           'insight', 'legend', 'mobilio', 'nsx', 'odyssey', 'passport',
           'pilot', 'ridgeline', 's660', 'vezel')

# -----------------------------
# Visualization Function
# -----------------------------
def view_classification(image, probabilities):
    """
    image: torch.Tensor of shape [3, H, W]
    probabilities: torch.Tensor of shape [num_classes]
    """
    # Convert to numpy for plotting
    ps = probabilities.data.cpu().numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(10, 5), ncols=2)

    # --- FIX 1: Correct Denormalization for Display ---
    # We must undo the ImageNet normalization so the image looks natural to the human eye
    # image = image * std + mean
    img = image.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1) # Ensure values stay valid between 0 and 1

    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title("Input Image")

    # --- Bar plot ---
    y_pos = np.arange(len(classes))
    ax2.barh(y_pos, ps, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(classes)
    ax2.invert_yaxis()  # labels read top-to-bottom
    ax2.set_xlabel('Probability')
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
    plt.show()

# -----------------------------
# Single image inference
# -----------------------------

# Load image
image_path = "/content/drive/MyDrive/Testcases_Honda/odyssey/odyssey.jpg"
image = Image.open(image_path).convert("RGB")

# --- FIX 2: Match Training Transforms Exactly ---
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Fixed typo: 244 -> 224
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

tensor_image = transform(image)
batched_image = tensor_image.unsqueeze(0).to(device)

# Inference
net.eval()
with torch.no_grad():
    logits = net(batched_image)

    # --- FIX 3: Use Softmax, not Exp ---
    # ResNet outputs raw 'logits'. We need Softmax to get percentages.
    probabilities = F.softmax(logits, dim=1).squeeze()

# Display
view_classification(tensor_image, probabilities)
