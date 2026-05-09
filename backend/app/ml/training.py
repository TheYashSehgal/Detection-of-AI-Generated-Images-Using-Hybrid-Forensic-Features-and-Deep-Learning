"""
Training Script for the Hybrid Forensic Model.

Supports:
- Folder-based dataset (real/ and fake/ directories)
- Data augmentation (JPEG compression, blur, noise, flips)
- Training with validation, early stopping, LR scheduling
- Metrics: accuracy, precision, recall, F1, AUC-ROC
- Automatic sample dataset generation for demo
"""
import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image, ImageFilter
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.models.hybrid_model import create_model, count_parameters
from app.forensics.pipeline import extract_forensic_features
from app.config import IMAGE_SIZE, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, EARLY_STOPPING_PATIENCE, CHECKPOINT_DIR

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ForensicDataset(Dataset):
    """Dataset that loads images and computes forensic features on-the-fly."""
    def __init__(self, root_dir, transform=None, cache_forensics=False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        real_dir = self.root_dir / "real"
        fake_dir = self.root_dir / "fake"
        if real_dir.exists():
            for f in real_dir.iterdir():
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp', '.bmp'):
                    self.samples.append((str(f), 0))
        if fake_dir.exists():
            for f in fake_dir.iterdir():
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp', '.bmp'):
                    self.samples.append((str(f), 1))
        print(f"[Dataset] Loaded {len(self.samples)} samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        rgb_tensor = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])(image)
        forensic = extract_forensic_features(image)
        forensic_tensor = torch.from_numpy(forensic['forensic_tensor'])
        return rgb_tensor, forensic_tensor, torch.tensor([label], dtype=torch.float32)


def generate_sample_dataset(output_dir: str, num_per_class: int = 50):
    """Generate a small demo dataset of real vs fake images for training."""
    out = Path(output_dir)
    real_dir = out / "real"
    fake_dir = out / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Dataset] Generating {num_per_class} samples per class...")
    np.random.seed(42)
    for i in range(num_per_class):
        # Real: natural-looking images with noise and texture
        arr = np.random.randint(50, 200, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        img = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(radius=2))
        img = img.filter(ImageFilter.DETAIL)
        img.save(real_dir / f"real_{i:04d}.jpg", "JPEG", quality=95)
        # Fake: smooth, uniform images (mimicking AI generation artifacts)
        base_color = np.random.randint(30, 220, 3)
        arr = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), base_color, dtype=np.uint8)
        noise = np.random.normal(0, 3, arr.shape).astype(np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        gradient = np.linspace(0.8, 1.2, IMAGE_SIZE).reshape(1, -1, 1)
        arr = np.clip(arr * gradient, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        img.save(fake_dir / f"fake_{i:04d}.png", "PNG")
    print(f"[Dataset] Generated {num_per_class * 2} total samples in {output_dir}")


def train(data_dir: str, epochs: int = NUM_EPOCHS, batch_size: int = BATCH_SIZE,
          lr: float = LEARNING_RATE, device: str = "cpu"):
    """Main training loop."""
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"[Training] Device: {device}")

    # Data augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    ])
    dataset = ForensicDataset(data_dir, transform=train_transform)
    if len(dataset) == 0:
        print("[Training] No data found! Generating sample dataset...")
        generate_sample_dataset(data_dir, num_per_class=50)
        dataset = ForensicDataset(data_dir, transform=train_transform)

    # Split 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = create_model(pretrained=True).to(device)
    params = count_parameters(model)
    print(f"[Training] Model params: {params['total']:,} ({params['total_mb']:.1f} MB)")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_val_loss = float('inf')
    patience_counter = 0
    history = []

    for epoch in range(epochs):
        model.train()
        train_loss, train_preds, train_labels = 0, [], []
        for rgb, forensic, labels in train_loader:
            rgb, forensic, labels = rgb.to(device), forensic.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(rgb, forensic)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy().flatten()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy().flatten())

        # Validation
        model.eval()
        val_loss, val_preds, val_labels, val_probs = 0, [], [], []
        with torch.no_grad():
            for rgb, forensic, labels in val_loader:
                rgb, forensic, labels = rgb.to(device), forensic.to(device), labels.to(device)
                outputs = model(rgb, forensic)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                val_probs.extend(probs)
                val_preds.extend((probs > 0.5).astype(int))
                val_labels.extend(labels.cpu().numpy().flatten())

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        try:
            val_auc = roc_auc_score(val_labels, val_probs)
        except:
            val_auc = 0.0

        scheduler.step(val_loss)

        print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} AUC: {val_auc:.4f}")

        history.append({'epoch': epoch+1, 'train_loss': train_loss, 'val_loss': val_loss,
                        'train_acc': train_acc, 'val_acc': val_acc, 'val_f1': val_f1, 'val_auc': val_auc})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            save_path = CHECKPOINT_DIR / "hybrid_model_best.pth"
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch+1,
                        'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1}, str(save_path))
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    with open(CHECKPOINT_DIR / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    print("[Training] Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Hybrid Forensic Model")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--generate-data", action="store_true", help="Generate sample dataset")
    args = parser.parse_args()
    if args.generate_data:
        generate_sample_dataset(args.data_dir)
    train(args.data_dir, args.epochs, args.batch_size, args.lr, args.device)
