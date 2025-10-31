# AutoNerve

AutoNerve is an image-classification project that uses a CNN + ViT hybrid model (PyTorch) to classify car brands and models. It includes training and evaluation code and a Streamlit-based web GUI for easy inference and demo.

Key points
- Task: Fine-grained car brand and model classification
- Model: CNN + Vision Transformer (ViT) hybrid implemented in PyTorch
- Interface: Streamlit web GUI for uploading images and viewing predictions
- License: Apache License 2.0 (see LICENSE)

Features
- Trainable PyTorch model combining convolutional feature extractor and ViT head
- Scripts for training, validation, and inference
- Streamlit web app for interactive predictions
- Support for saving/loading model checkpoints and label mappings

Quick start (example)
1. Create virtual environment and install dependencies:
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt

   If you don't have a requirements.txt, install core packages:
   pip install torch torchvision timm streamlit numpy pandas pillow scikit-learn tqdm

2. Prepare dataset (see Dataset section).

3. Train (example):
   python train.py --data /path/to/data --epochs 30 --batch-size 32 --outdir checkpoints/

4. Run Streamlit GUI (example):
   streamlit run app.py
   Then open the shown local URL (e.g., http://localhost:8501) to upload images and get predictions.

Project structure (suggested / common)
- data/                    # dataset storage / example split
- notebooks/               # EDA and experiments
- src/ or models/          # model definitions (CNN & ViT hybrid)
- train.py                 # training script
- evaluate.py              # evaluation / metrics
- infer.py                 # CLI inference script
- app.py                   # Streamlit app for demo / GUI
- checkpoints/             # saved models
- requirements.txt
- LICENSE

Model & architecture
- Backbone: convolutional network to extract local features (e.g., ResNet or custom CNN)
- Head: Vision Transformer (ViT) that operates on patch/feature tokens produced by the CNN to learn global relationships
- Loss: Cross-entropy for multi-class classification (optionally label smoothing)
- Optimizer & schedulers: Recommend AdamW + cosine or step LR schedule
- Data augmentation: Random crops, flips, color jitter, normalization (important for fine-grained car classification)

Dataset format
- Directory-of-folders (ImageFolder style):
  data/
    train/
      brand_model_1/
        img1.jpg
        img2.jpg
      brand_model_2/
    val/
      brand_model_1/
      brand_model_2/
- Or a CSV file with columns: filepath,label
- Save a label map (class index → class name) after training for inference

Training tips
- Start with smaller image sizes (224x224) and increase if GPU allows
- Use transfer learning: initialize CNN and ViT from pretrained weights (timm)
- Monitor validation accuracy and confusion matrix to identify confusions between similar models
- Use class-balanced sampling or loss if classes are heavily imbalanced

Example training command (replace args with your script's flags)
python train.py \
  --data data/ \
  --train-dir train \
  --val-dir val \
  --arch cnn_vit_hybrid \
  --pretrained \
  --epochs 40 \
  --batch-size 16 \
  --lr 1e-4 \
  --checkpoint-dir checkpoints/

Inference (CLI)
- Single image:
  python infer.py --image /path/to/car.jpg --checkpoint checkpoints/best.pth --labels label_map.json
- Batch folder:
  python infer.py --folder /path/to/images --checkpoint checkpoints/best.pth --out predictions.csv

Streamlit demo (app.py)
- Typical behavior:
  - Upload image or pick sample
  - Model predicts top-k classes with probabilities
  - Shows bounding/resized preview and class label
- Run:
  streamlit run app.py

Saving & loading models
- Save model.state_dict(), optimizer state, epoch, and label map as a checkpoint (e.g., torch.save(dict(...), PATH))
- Load with model.load_state_dict(...) and set model.eval() for inference

Evaluation & metrics
- Top-1 and Top-5 accuracy
- Confusion matrix for brand/model-level analysis
- Precision/recall per class (helpful for imbalanced classes)

Logging & experiments
- Use TensorBoard or Weights & Biases for experiment tracking:
  - TensorBoard: add SummaryWriter logs and run tensorboard --logdir runs/
  - W&B: wandb.init(...) and log metrics and model checkpoints

Docker (optional)
- Example Dockerfile steps:
  - FROM python:3.9
  - COPY . /app
  - WORKDIR /app
  - RUN pip install -r requirements.txt
  - CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

Common pitfalls and troubleshooting
- Out of memory: reduce batch size or image size
- Wrong label mapping: ensure label_map used at inference matches training
- Poor accuracy: check dataset labeling, augmentation, and learning rate
- Streamlit not loading: ensure correct app.py path and that Streamlit is installed

Contributing
- Fork the repo
- Create a branch: git checkout -b feature/my-feature
- Add tests where applicable and update documentation
- Submit a PR with a clear description and reproducible steps

License
This project is licensed under the Apache License 2.0. See LICENSE for details.

Contact
Owner: waqarbaig6133
Repo: https://github.com/waqarbaig6133/AutoNerve

If you want, I can:
- Generate a ready-to-commit README.md file and (optionally) create/commit it to your repo.
- Inspect the repository to fill in exact script names, arguments, and real dependency list and then produce a tailored README. Which would you prefer?  

— GitHub Copilot Chat Assistant