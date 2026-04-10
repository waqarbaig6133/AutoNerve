# AutoNerve

**AutoNerve** is a deep learning project for fine-grained Honda car model classification. It uses a pretrained **ResNet-18** backbone fine-tuned with two-stage transfer learning to classify car images into 17 Honda model categories.

---

## Supported Classes

`Accord` · `Amaze` · `Brio` · `City` · `Civic` · `Clarity` · `Freed` · `Insight` · `Legend` · `Mobilio` · `NSX` · `Odyssey` · `Passport` · `Pilot` · `Ridgeline` · `S660` · `Vezel`

---

## Project Structure

```
AutoNerve/
└── TestSingleBrand/
    ├── Version-0.1-Honda.py  # Training script
    ├── ModelTest.py          # Inference + visualization
    └── SaveState.py          # Checkpoint saving
```

---

## Installation

```bash
git clone https://github.com/waqarbaig6133/AutoNerve.git
cd AutoNerve

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install torch torchvision numpy pandas pillow matplotlib
```

---

## Training

Training runs on **Google Colab** with the dataset stored in Google Drive.

**1. Upload your image dataset to Google Drive** in `ImageFolder` format:

```
honda_cars/
├── accord/
│   ├── img1.jpg
│   └── img2.jpg
├── civic/
└── ...
```

**2. Open `TestSingleBrand/Version-0.1-Honda.py`** in Colab and run it. The script:

- Mounts Google Drive and loads images from `/content/drive/MyDrive/honda_cars`
- Applies ImageNet normalization and random horizontal flip augmentation
- Splits 10% of data for validation automatically
- **Stage 1** (5 epochs): trains only the final fully-connected layer (lr = 1e-3)
- **Stage 2** (25 epochs): fine-tunes the entire ResNet-18 (lr = 1e-4)

**3. Save the checkpoint** with `TestSingleBrand/SaveState.py`:

```python
torch.save({
    "model_state": net.state_dict(),
    "class_names": class_names
}, "/content/drive/MyDrive/honda_resnet18.pth")
```

---

## Inference

Run `TestSingleBrand/ModelTest.py` to classify a single image and visualize class probabilities. Update the image path at the top of the script:

```python
image_path = "/path/to/your/car.jpg"
```

The script outputs the input image alongside a bar chart of per-class softmax probabilities. Top-K predictions are also available via the `predict_image` helper:

```python
top_class, confidence, other_guesses = predict_image(image_path, net, device, top_k=3)
print(f"Prediction: {top_class} ({confidence:.1%} confidence)")
```

---

## Requirements

| Package | Purpose |
|---|---|
| `torch` / `torchvision` | Model training and inference |
| `Pillow` | Image loading |
| `matplotlib` | Probability visualization |
| `numpy` | Array operations |

---

## License

This project is licensed under the **Apache License 2.0**. See [LICENSE](LICENSE) for details.

---

## Author

**waqarbaig6133** — [github.com/waqarbaig6133/AutoNerve](https://github.com/waqarbaig6133/AutoNerve)
