# 🎭 Deep Learning Based Face Mask & Face Spoof Detection System

[![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> An end-to-end deep learning system that detects face masks and face spoofing attempts in real time, classifying faces into **Real**, **Spoofed**, or **Masked** categories using a ResNet18-based architecture with MTCNN face detection.

---

## 📌 About This Project

With the growing need for secure facial authentication systems, detecting both **face spoofing attacks** (photo, screen, or print attacks) and **masked faces** is critical for reliable biometric security.

This project builds a production-ready deep learning pipeline that:
- Detects and crops faces using **MTCNN**
- Classifies them using a fine-tuned **ResNet18** model
- Serves predictions via a **FastAPI REST API**
- Is fully containerized with **Docker** for easy deployment

---

## 🏷️ Classification Categories

| Class | Description |
|---|---|
| ✅ **Real Face** | Genuine live face |
| 🚫 **Spoofed Face** | Photo / screen / print attack |
| 😷 **Masked Face** | Face with a mask on |

---

## 📁 Repository Structure

```
Deep-Learning-Based-Face-Mask-Face-Spoof-Detection-System/
├── app.py                    # FastAPI REST API — /predict endpoint
├── model.py                  # ResNet18-based FaceSpoofMaskClassifier
├── train.py                  # Model training pipeline
├── evaluate.py               # Evaluation — classification report & confusion matrix
├── face_detection.py         # MTCNN face detection & cropping
├── data_preprocessing.py     # Dataset preparation & train/val/test split
├── Dockerfile                # Docker container configuration
├── __init__.py
├── .gitkeep
└── README.md
```

---

## 🧠 Model Architecture

**Base Model:** ResNet18 (pretrained on ImageNet)

The final fully connected layer is replaced with a custom classification head:

```
ResNet18 Backbone
    └── Linear(512) → ReLU → Dropout(0.5) → Linear(3 classes)
```

- **Input:** 224×224 RGB face image (normalized)
- **Output:** Probabilities for 3 classes — real, spoof, masked
- **Device:** Automatically uses GPU (CUDA) if available, else CPU

---

## ⚙️ Pipeline Overview

### 1. Data Preprocessing (`data_preprocessing.py`)
- Combines 3 datasets: Face Mask 12K, CelebA-Spoof, Anti-Spoofing
- Splits data into **Train (70%) / Val (15%) / Test (15%)**
- Organizes into class folders: `real/`, `spoof/`, `masked/`

### 2. Face Detection (`face_detection.py`)
- Uses **MTCNN** to detect and crop the largest face in each image
- Adds a 20% margin around detected faces for context
- Falls back to full image resize if no face is detected

### 3. Model Training (`train.py`)
- **Batch size:** 32 | **Epochs:** 20 | **Learning rate:** 0.001
- **Data augmentation:** Random horizontal flip, rotation (±10°), color jitter
- Saves best model to `models/best_model.pth`

### 4. Evaluation (`evaluate.py`)
- Generates full classification report (Precision, Recall, F1-Score)
- Plots confusion matrix with Seaborn heatmap
- Saves results to `results/` directory

### 5. REST API (`app.py`)
- Built with **FastAPI**
- `POST /predict` — upload an image, get back predicted class + confidence
- Interactive docs available at `/docs`

---

## 🌐 API Usage

### Run Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn app:app --reload
```

### Test the API
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -F "file=@your_image.jpg"
```

### Example Response
```json
{
  "prediction": "real",
  "confidence": 0.97,
  "all_scores": {
    "real": 0.97,
    "spoof": 0.02,
    "masked": 0.01
  }
}
```

---

## 🐳 Docker Deployment

```bash
# Build the Docker image
docker build -t face-spoof-mask-detector .

# Run the container
docker run -p 8000:8000 face-spoof-mask-detector
```

---

## 🛠️ Tech Stack

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![ResNet18](https://img.shields.io/badge/ResNet18-Transfer_Learning-blueviolet?style=flat-square)
![MTCNN](https://img.shields.io/badge/MTCNN-Face_Detection-ff69b4?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C9BE8?style=flat-square)

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/suchismittaa/Deep-Learning-Based-Face-Mask-Face-Spoof-Detection-System.git
cd Deep-Learning-Based-Face-Mask-Face-Spoof-Detection-System

# Install dependencies
pip install -r requirements.txt

# Step 1: Prepare data
python data_preprocessing.py

# Step 2: Train the model
python train.py

# Step 3: Evaluate the model
python evaluate.py

# Step 4: Launch the API
uvicorn app:app --reload
```

---

## 📬 Connect with Me

**Suchismita Sarkar** — BTech IT, KIIT Bhubaneswar (CGPA: 8.11)

[![Portfolio](https://img.shields.io/badge/🌐_Portfolio-Visit_Now-4f46e5?style=for-the-badge)](https://suchismitasportfolio.netlify.app/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/suchismitasarkar222/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github)](https://github.com/suchismittaa)

---

## 📄 License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute this code, provided that proper credit is given to the original author.

See the [LICENSE](LICENSE) file for full details.

---

<p align="center">
  Made with ❤️ for safer facial authentication — Suchismita Sarkar
</p>
