"""
api/main.py
──────────────────────────────────────────────────────────────────────────────
FastAPI backend — exposes two endpoints consumed by the Streamlit frontend.

Endpoints
─────────
POST /train
    Body (JSON):
        {
            "model_name"   : "U-Net" | "ResNet" | "Inception",
            "learning_rate": 1e-3,
            "epochs"       : 10,
            "batch_size"   : 32,
            "dropout_rate" : 0.5,
            "image_size"   : 224
        }
    Response (JSON):
        {
            "train_losses" : [0.72, 0.55, ...],
            "val_losses"   : [0.80, 0.61, ...],
            "train_accs"   : [0.61, 0.74, ...],
            "val_accs"     : [0.55, 0.70, ...],
            "train_aucs"   : [...],
            "val_aucs"     : [...],
            "best_val_auc" : 0.94
        }

POST /predict
    Body (JSON):
        {
            "model_name": "U-Net" | "ResNet" | "Inception",
            "image_size": 224
        }
    Response (JSON):
        {
            "predictions": [
                {"id": "img_0001", "prediction": 0.91},
                {"id": "img_0002", "prediction": 0.07},
                ...
            ]
        }

Run with:
    uvicorn main:app --reload --port 8000
"""

import sys
from pathlib import Path

# Allow importing from the project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from models import MODEL_REGISTRY


app = FastAPI(title="Pneumonia Detection API")

# ── Paths ──────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).resolve().parent.parent
TRAIN_DIR   = ROOT / "data" / "train"
VAL_DIR     = ROOT / "data" / "val"
TEST_DIR    = ROOT / "data" / "test_for_students"
WEIGHTS_DIR = ROOT / "api" / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of TTA passes at inference (higher = better AUC, slower)
N_TTA = 8


# ── Request / Response schemas ─────────────────────────────────────────────────

class TrainRequest(BaseModel):
    model_name:    str   = Field(..., description="U-Net | ResNet | Inception")
    learning_rate: float = Field(3e-4,  ge=1e-6, le=1.0)
    epochs:        int   = Field(10,    ge=1,    le=100)
    batch_size:    int   = Field(32,    ge=4,    le=128)
    dropout_rate:  float = Field(0.3,   ge=0.0,  le=0.9)
    image_size:    int   = Field(224,   ge=64,   le=512)


class TrainResponse(BaseModel):
    train_losses: list[float]
    val_losses:   list[float]
    train_accs:   list[float]
    val_accs:     list[float]
    train_aucs:   list[float]
    val_aucs:     list[float]
    best_val_auc: float


class PredictRequest(BaseModel):
    model_name: str = Field(..., description="U-Net | ResNet | Inception")
    image_size: int = Field(224, ge=64, le=512)


class PredictionItem(BaseModel):
    id:         str
    prediction: float


class PredictResponse(BaseModel):
    predictions: list[PredictionItem]


# ── Loss ───────────────────────────────────────────────────────────────────────

class LabelSmoothingBCE(nn.Module):
    """BCEWithLogitsLoss + label smoothing to reduce overconfidence."""
    def __init__(self, pos_weight: torch.Tensor | None = None, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets)


# ── Data helpers ───────────────────────────────────────────────────────────────

def get_transforms(image_size: int, augment: bool = False):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if augment:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])


def get_tta_transforms(image_size: int):
    """Light augmentations applied repeatedly at inference (TTA)."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize,
    ])


def get_dataloaders(image_size: int, batch_size: int):
    """Build train and val DataLoaders from ImageFolder structure."""
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=get_transforms(image_size, augment=True))
    val_dataset   = datasets.ImageFolder(VAL_DIR,   transform=get_transforms(image_size, augment=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, train_dataset


def compute_pos_weight(train_dataset) -> torch.Tensor:
    """pos_weight = n_negative / n_positive for BCEWithLogitsLoss."""
    counts = np.bincount([label for _, label in train_dataset.samples])
    return torch.tensor([counts[0] / counts[1]], dtype=torch.float32).to(DEVICE)


# ── Training helpers ───────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs.flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = float(np.mean((np.array(all_probs) >= 0.5) == np.array(all_labels)))
    auc      = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    return avg_loss, accuracy, auc


def evaluate(model, loader, criterion) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            logits = model(images)
            loss   = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = float(np.mean((np.array(all_probs) >= 0.5) == np.array(all_labels)))
    auc      = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    return avg_loss, accuracy, auc


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    if req.model_name not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown model '{req.model_name}'. Choose from {list(MODEL_REGISTRY)}")

    train_loader, val_loader, train_dataset = get_dataloaders(req.image_size, req.batch_size)

    model      = MODEL_REGISTRY[req.model_name](in_channels=3, dropout_rate=req.dropout_rate).to(DEVICE)
    pos_weight = compute_pos_weight(train_dataset)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=req.learning_rate, weight_decay=1e-4)
    criterion  = LabelSmoothingBCE(pos_weight=pos_weight, smoothing=0.1)

    # 3-epoch linear warmup then cosine decay
    warmup_epochs = min(3, req.epochs)
    warmup    = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine    = CosineAnnealingLR(optimizer, T_max=max(req.epochs - warmup_epochs, 1), eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    train_aucs,   val_aucs   = [], []
    best_val_auc = 0.0

    for epoch in range(req.epochs):
        tr_loss, tr_acc, tr_auc = train_one_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc, vl_auc = evaluate(model, val_loader, criterion)

        train_losses.append(tr_loss);  val_losses.append(vl_loss)
        train_accs.append(tr_acc);     val_accs.append(vl_acc)
        train_aucs.append(tr_auc);     val_aucs.append(vl_auc)

        if vl_auc > best_val_auc:
            best_val_auc = vl_auc
            weight_path  = WEIGHTS_DIR / f"{req.model_name.replace(' ', '_')}_best.pt"
            torch.save(model.state_dict(), weight_path)

        print(f"Epoch {epoch+1}/{req.epochs} | lr={optimizer.param_groups[0]['lr']:.2e} | "
              f"Train loss {tr_loss:.4f} acc {tr_acc:.4f} auc {tr_auc:.4f} | "
              f"Val   loss {vl_loss:.4f} acc {vl_acc:.4f} auc {vl_auc:.4f}")

        scheduler.step()

    return TrainResponse(
        train_losses=train_losses,
        val_losses=val_losses,
        train_accs=train_accs,
        val_accs=val_accs,
        train_aucs=train_aucs,
        val_aucs=val_aucs,
        best_val_auc=best_val_auc,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.model_name not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown model '{req.model_name}'")

    weight_path = WEIGHTS_DIR / f"{req.model_name.replace(' ', '_')}_best.pt"
    if not weight_path.exists():
        raise HTTPException(status_code=404, detail=f"No saved weights found for '{req.model_name}'. Train it first.")

    model = MODEL_REGISTRY[req.model_name](in_channels=3).to(DEVICE)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.eval()

    tf_base = get_transforms(req.image_size, augment=False)
    tf_tta  = get_tta_transforms(req.image_size)

    image_paths = (sorted(TEST_DIR.glob("*.jpeg")) +
                   sorted(TEST_DIR.glob("*.jpg"))  +
                   sorted(TEST_DIR.glob("*.png")))
    if not image_paths:
        raise HTTPException(status_code=404, detail=f"No images found in {TEST_DIR}")

    from PIL import Image

    predictions = []
    with torch.no_grad():
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")

            # base pass + N_TTA augmented passes → average
            probs = [torch.sigmoid(model(tf_base(img).unsqueeze(0).to(DEVICE))).item()]
            for _ in range(N_TTA):
                probs.append(torch.sigmoid(model(tf_tta(img).unsqueeze(0).to(DEVICE))).item())

            predictions.append(PredictionItem(
                id=img_path.stem,
                prediction=round(float(np.mean(probs)), 6),
            ))

    return PredictResponse(predictions=predictions)


# ── Health check ───────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "device": str(DEVICE), "models": list(MODEL_REGISTRY.keys())}
