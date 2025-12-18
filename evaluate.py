import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import sys
import cv2
from tqdm import tqdm

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import FaceSpoofMaskClassifier
from src.face_detection import FaceDetector

# --- CONFIGURATION ---
DATA_DIR = 'data/processed'
RESULTS_DIR = 'results'
CLASSES = ['real', 'spoof', 'masked']

# Re-use the FaceDataset class from train.py
class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = torch.utils.data.datasets.ImageFolder(root_dir)
        self.transform = transform
        self.face_detector = FaceDetector()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, label = self.dataset.samples[idx]
        image = self.dataset.loader(img_path)
        image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        cropped_face = self.face_detector.detect_and_crop(image_rgb)
        
        if cropped_face is None:
            cropped_face = cv2.resize(image_rgb, (224, 224))

        if self.transform:
            cropped_face = self.transform(cropped_face)
            
        return cropped_face, label

def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    return all_labels, all_preds

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = FaceSpoofMaskClassifier(num_classes=3).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # Prepare data
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = FaceDataset(os.path.join(DATA_DIR, 'test'), transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Evaluate
    labels, preds = evaluate(model, test_loader, device)
    
    # Generate and save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Classification Report
    report = classification_report(labels, preds, target_names=CLASSES)
    with open(os.path.join(RESULTS_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)
    print("Classification Report:\n", report)
        
    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
    print(f"Confusion matrix saved to {os.path.join(RESULTS_DIR, 'confusion_matrix.png')}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the Face Spoof & Mask Detection Model')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth', help='Path to the trained model file')
    args = parser.parse_args()
    main(args)