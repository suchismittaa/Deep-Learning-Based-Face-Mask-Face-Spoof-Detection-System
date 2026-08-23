"""Training pipeline for the Real / Spoof / Masked face classifier.

Run:
    python train.py
"""
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import FaceSpoofMaskClassifier
from src.face_detection import FaceDetector

DATA_DIR = "data/processed"
MODEL_SAVE_PATH = "models/best_model.pth"
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
NUM_WORKERS = 0  # safer default across Windows/macOS/Linux
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED):
    """Make dataset splitting/augmentation/training behaviour reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FaceDataset(Dataset):
    """Wraps torchvision.ImageFolder and runs MTCNN face detection/cropping
    per sample so that training sees exactly the same kind of crop the
    inference API produces (see app.py's /predict handler)."""

    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root_dir)
        self.transform = transform
        self.face_detector = FaceDetector()

    @property
    def class_to_idx(self):
        return self.dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, label = self.dataset.samples[idx]
        # PIL already decodes to RGB - no BGR conversion needed here.
        image = self.dataset.loader(img_path).convert("RGB")
        image_rgb = np.array(image)
        cropped_face = self.face_detector.detect_and_crop(image_rgb)
        if cropped_face is None:
            # No face found (rare with curated datasets) - fall back to a
            # plain resize rather than dropping the sample.
            cropped_face = cv2.resize(image_rgb, (224, 224), interpolation=cv2.INTER_AREA)
        if self.transform:
            cropped_face = self.transform(cropped_face)
        return cropped_face, label


train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def compute_class_weights(dataset: FaceDataset) -> torch.Tensor:
    """Inverse-frequency class weights, used because the three source
    datasets are not guaranteed to contribute equal counts per class.
    """
    counts = np.bincount(
        [label for _, label in dataset.dataset.samples],
        minlength=len(dataset.class_to_idx),
    )
    counts = np.maximum(counts, 1)  # avoid division by zero for an empty class
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)
    total_loss, total_items = 0.0, 0
    correct = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for inputs, labels in tqdm(loader, desc="Training" if is_training else "Validation"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            if is_training:
                optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_items += inputs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

    return total_loss / max(total_items, 1), correct / max(total_items, 1)


def main():
    set_seed(SEED)
    print(f"Using device: {DEVICE}")

    train_dataset = FaceDataset(os.path.join(DATA_DIR, "train"), train_transform)
    val_dataset = FaceDataset(os.path.join(DATA_DIR, "val"), val_transform)

    # IMPORTANT: torchvision.ImageFolder assigns class indices alphabetically
    # by folder name (masked=0, real=1, spoof=2), NOT in the order classes
    # happen to be listed anywhere else in this project. We save this
    # mapping into the checkpoint so app.py/evaluate.py can never get the
    # label order wrong, instead of hardcoding an assumed order.
    class_to_idx = train_dataset.class_to_idx
    print(f"Class to index mapping: {class_to_idx}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = FaceSpoofMaskClassifier(num_classes=len(class_to_idx), pretrained=True).to(DEVICE)

    class_weights = compute_class_weights(train_dataset).to(DEVICE)
    print(f"Class weights (inverse frequency): {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)

        print(
            f"Train loss: {train_loss:.4f} | Train accuracy: {train_acc:.3f} | "
            f"Val loss: {val_loss:.4f} | Val accuracy: {val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "epoch": epoch + 1,
                },
                MODEL_SAVE_PATH,
            )
            print(f"Saved best model -> {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
