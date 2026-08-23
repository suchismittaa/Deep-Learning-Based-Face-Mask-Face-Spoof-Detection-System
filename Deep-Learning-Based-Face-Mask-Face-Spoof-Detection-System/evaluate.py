"""Evaluate a trained checkpoint on the held-out test split.

Run:
    python evaluate.py --model_path models/best_model.pth

Writes results/classification_report.txt and results/confusion_matrix.png.
Nothing here is fabricated: if you have not trained a model yet, run
train.py first - this script will refuse to invent numbers.
"""
import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import FaceSpoofMaskClassifier
from src.face_detection import FaceDetector

DATA_DIR = "data/processed"
RESULTS_DIR = "results"

eval_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class FaceDataset(Dataset):
    """Same detect-crop-normalize pipeline used at training/inference time,
    applied here to the held-out test split so evaluation numbers reflect
    real inference-time behaviour."""

    def __init__(self, root_dir, transform=None):
        self.dataset = datasets.ImageFolder(root_dir)
        self.transform = transform
        self.face_detector = FaceDetector()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, label = self.dataset.samples[idx]
        image = self.dataset.loader(img_path).convert("RGB")
        image_rgb = np.array(image)
        cropped_face = self.face_detector.detect_and_crop(image_rgb)
        if cropped_face is None:
            cropped_face = cv2.resize(image_rgb, (224, 224), interpolation=cv2.INTER_AREA)
        return self.transform(cropped_face), label


def main(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        print(f"No checkpoint found at {model_path}.")
        print("Run `python train.py` first - evaluation results are not fabricated.")
        return

    checkpoint = torch.load(model_path, map_location=device)
    class_to_idx = checkpoint["class_to_idx"]
    # Sort by index so label i in the report matches predicted index i.
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]

    model = FaceSpoofMaskClassifier(num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_dir = os.path.join(DATA_DIR, "test")
    if not os.path.isdir(test_dir) or not os.listdir(test_dir):
        print(f"No test data found at {test_dir}. Run data_preprocessing.py first.")
        return

    test_dataset = FaceDataset(test_dir, eval_transform)
    # Sanity check: the test set's own class_to_idx must match the one the
    # model was trained with, or the report would be silently mislabeled.
    if test_dataset.dataset.class_to_idx != class_to_idx:
        print("Warning: test set class_to_idx does not match the checkpoint's. "
              f"checkpoint={class_to_idx} test={test_dataset.dataset.class_to_idx}")

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    labels, preds = [], []
    with torch.no_grad():
        for inputs, batch_labels in test_loader:
            outputs = model(inputs.to(device))
            preds.extend(outputs.argmax(1).cpu().numpy())
            labels.extend(batch_labels.numpy())

    os.makedirs(RESULTS_DIR, exist_ok=True)
    report = classification_report(labels, preds, target_names=class_names, digits=4)
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(labels, preds, labels=range(len(class_names)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=180)
    plt.close()

    print(report)
    print(f"Saved results to {RESULTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Face Spoof & Mask Detection model")
    parser.add_argument("--model_path", type=str, default="models/best_model.pth", help="Path to the trained checkpoint")
    args = parser.parse_args()
    main(args.model_path)
