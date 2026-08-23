# Face Mask & Face Spoof Detection System

A computer-vision pipeline that classifies a face image as **Real**, **Spoof**, or **Masked** using **MTCNN** face detection and a **ResNet18** transfer-learning classifier, served through a **FastAPI** inference API with an interactive frontend case study.

## Project Overview

Biometric face-based authentication is only as trustworthy as its ability to reject presentation attacks (a printed photo, a phone/laptop screen held up to the camera) and to correctly handle everyday cases like a person wearing a mask. This project builds and serves a single classifier that tells the three cases apart, and packages the full pipeline — data, face detection, training, evaluation, and a demo API — as a reviewable portfolio project.

## Problem Statement

A face-verification system that only checks "does this face match?" can be fooled by a photo of the enrolled user, and can also misbehave when the user is legitimately wearing a mask. This project frames both concerns as a single 3-way classification problem — **real / spoof / masked** — so a downstream system can decide how to respond to each case instead of only getting a binary match/no-match signal.

## Features

- MTCNN-based face detection and cropping with a contextual margin
- ResNet18 transfer-learning classifier with a custom 3-class head
- Reproducible data preprocessing and train/val/test split
- Class-weighted training loss to account for dataset imbalance
- FastAPI `/predict` endpoint returning class + per-class probabilities
- `/health` endpoint for deployment checks
- Evaluation script producing a classification report and confusion matrix
- Dockerfile for containerized deployment
- Self-contained, dependency-free interactive frontend (`frontend/index.html`)

## Classification Classes

| Class | Meaning |
|---|---|
| **Real** | A genuine, live face captured directly by the camera. |
| **Spoof** | A presentation attack — a photo, print, or screen replay of a face rather than a live person. |
| **Masked** | A live face wearing a mask. |

## Dataset Pipeline

`data_preprocessing.py` consolidates three source datasets into the three target classes, then performs a reproducible split:

| Source dataset | Contributes to |
|---|---|
| Face Mask 12K | `masked` (WithMask), `real` (WithoutMask) |
| CelebA-Spoof | `real` (live), `spoof` (spoof) |
| Anti-Spoofing dataset | `real`, `spoof` |

Pipeline steps:

1. Copy each source folder's images into staging folders `data/processed/{real,spoof,masked}`, using `shutil.copy2` (never move) so the original raw datasets are left untouched.
2. Shuffle each class with a fixed random seed (`42`) and split **70% train / 15% validation / 15% test**.
3. Move the split images into `data/processed/{train,val,test}/{class}`.
4. If two source datasets happen to share a filename, the collision is detected and the incoming file is renamed instead of silently overwriting the original (see [Important Issues Discovered](#important-issues-discovered-in-the-original-project)).

These ratios and this pipeline only reflect what `data_preprocessing.py` actually does — nothing here is aspirational.

## Face Detection

`src/face_detection.py` wraps MTCNN:

1. Run MTCNN on the RGB input image.
2. If multiple faces are detected, keep the one with the largest bounding-box area.
3. Expand the box by a 20% margin on each side for context.
4. Clip the box to the image boundaries and reject degenerate (zero-area) crops.
5. Return `None` if no face is found, so callers can fall back or reject the request explicitly instead of crashing.

## Model Architecture

**Base model:** ResNet18, pretrained on ImageNet, with the final fully-connected layer replaced by a custom head:

```
Input Face (224×224 RGB, ImageNet-normalized)
        │
   ResNet18 backbone (pretrained)
        │
     Linear(512)
        │
       ReLU
        │
    Dropout(0.5)
        │
   Linear(num_classes)
        │
  real / spoof / masked
```

**Why ResNet18?** It's small enough to fine-tune quickly on a modest dataset and CPU/single-GPU hardware, while its ImageNet-pretrained features already encode general visual structure (edges, textures, shapes) that transfers well to a face-classification task with a limited amount of task-specific data.

## Training

Configuration actually used in `train.py`:

| Setting | Value |
|---|---|
| Batch size | 32 |
| Epochs | 20 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Loss | Cross-entropy, class-weighted by inverse frequency |
| Augmentation | Random horizontal flip, ±10° rotation, brightness/contrast jitter |
| Seed | 42 |

```
   TRAIN
     │
  VALIDATE
     │
 COMPARE LOSS
     │
SAVE BEST MODEL  (models/best_model.pth, kept whenever val loss improves)
```

The checkpoint stores the model weights **and** the `class_to_idx` mapping produced by `ImageFolder`, so the class each output index corresponds to is never assumed — it's read back from the checkpoint at inference and evaluation time.

## Evaluation

`evaluate.py` loads a trained checkpoint, runs it over the held-out test split (using the exact same face-detection + preprocessing path as training and inference), and writes:

- `results/classification_report.txt` — precision, recall, F1-score, accuracy per class
- `results/confusion_matrix.png` — confusion matrix heatmap

**No numbers are included in this README or the frontend.** This repository does not ship a trained checkpoint or benchmark results — those depend on which raw datasets you provide locally. Run `python evaluate.py` after training to generate real results for your own run.

## API

Built with FastAPI (`app.py`):

| Endpoint | Description |
|---|---|
| `GET /health` | Reports whether the model is loaded, which device it's on, and the resolved class order. |
| `POST /predict` | Accepts an image file, runs detection + classification, returns the predicted class and per-class probabilities. |

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -F "file=@your_image.jpg"
```

Example response shape (values are illustrative only — actual numbers come from your trained model):

```json
{
  "prediction": "real",
  "confidence": 0.9123,
  "all_probabilities": {
    "masked": 0.0244,
    "real": 0.9123,
    "spoof": 0.0633
  }
}
```

Interactive docs are available at `http://127.0.0.1:8000/docs` once the server is running.

## Frontend

`frontend/index.html` is a single, dependency-free HTML/CSS/JS file — no build step, no framework. It explains the project (problem, data, pipeline, architecture, limitations) and includes a live demo panel that uploads an image to `POST /predict` and renders the actual JSON response, including an animated probability bar per class. If the API isn't running, the demo panel shows a clear connection-error state instead of fabricating a result.

## Project Structure

```
Face-Mask-Face-Spoof-Detection-System/
│
├── app.py                     # FastAPI inference API
├── train.py                   # Training pipeline
├── evaluate.py                # Test-set evaluation
├── data_preprocessing.py      # Dataset consolidation + train/val/test split
├── requirements.txt
├── Dockerfile
├── README.md
├── LICENSE
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── model.py                # ResNet18-based classifier
│   └── face_detection.py       # MTCNN detection + cropping
│
├── models/
│   └── best_model.pth          # created by train.py (not included)
│
├── data/
│   ├── raw/                    # place source datasets here (not included)
│   └── processed/               # created by data_preprocessing.py
│
├── results/                    # created by evaluate.py
│
└── frontend/
    └── index.html               # interactive case study + live demo
```

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. `mtcnn` pulls in `tensorflow` as a backend, so the first install may take a few minutes.

## Running Locally

### 1. Get the source datasets

This repository does not include the raw datasets or a trained checkpoint (large, and covered by their own dataset licenses). Download them yourself and place them at:

```
data/raw/face-mask-12k-images-dataset/
data/raw/celeba-spoof-for-face-antispoofing/
data/raw/anti-spoofing/
```

matching the folder layout `data_preprocessing.py` expects (see the source code for exact subfolder names).

### 2. Preprocess

```bash
python data_preprocessing.py
```

### 3. Train

```bash
python train.py
```

Saves the best checkpoint to `models/best_model.pth`.

### 4. Evaluate

```bash
python evaluate.py --model_path models/best_model.pth
```

### 5. Run the API

```bash
uvicorn app:app --reload
```

### 6. Open the frontend

Open `frontend/index.html` directly in a browser. The live demo panel calls `http://127.0.0.1:8000` by default — update the API base URL field in the page if your server runs elsewhere.

## Docker

```bash
docker build -t face-spoof-mask-detector .
docker run -p 8000:8000 -v $(pwd)/models:/app/models face-spoof-mask-detector
```

The volume mount makes a locally trained `models/best_model.pth` available inside the container without rebuilding the image.

## Important Issues Discovered in the Original Project

An honest audit of the original code surfaced several correctness bugs, fixed in this version:

- **Class-label ordering mismatch.** `torchvision.ImageFolder` assigns class indices alphabetically by folder name (`masked=0, real=1, spoof=2`), but the API hardcoded `["real", "spoof", "masked"]`. This would have silently mislabeled every prediction. Fixed by saving `class_to_idx` into the training checkpoint and reading the label order back from it at inference/evaluation time, instead of hardcoding an assumption.
- **Incorrect color-channel handling.** The original API and dataset code ran `cv2.cvtColor(image, COLOR_BGR2RGB)` on images that PIL had already decoded as RGB, which reverses the channel order instead of correcting it, weakening the benefit of ImageNet-pretrained weights (whose normalization stats assume true RGB order). Fixed by decoding with `Image.open(...).convert("RGB")` and dropping the unnecessary conversion.
- **Broken preprocessing directory structure.** `data_preprocessing.py` created `data/processed/{train,val,test}/{class}` folders but then tried to copy images directly into `data/processed/{class}` (which was never created), so preprocessing would fail before any split happened. Fixed by creating both the staging and split directory trees upfront.
- **Silent file overwrites during dataset merging.** Copying same-named files from multiple source datasets into one folder could overwrite an image from a different dataset without warning. Fixed by detecting name collisions and renaming the incoming file.
- **Non-reproducible train/val/test split.** `random.shuffle` was called without a fixed seed, so re-running preprocessing produced a different split every time (making results impossible to compare across runs). Fixed with a fixed seed (`42`).
- **Missing imports that would crash on first run.** `train.py` and `evaluate.py` referenced `datasets.ImageFolder` and `numpy` without importing either.
- **`evaluate.py` referenced a non-existent API** (`torch.utils.data.datasets.ImageFolder` instead of `torchvision.datasets.ImageFolder`).
- **No `/health` endpoint, no CORS.** The API had no way to check model status without a live prediction request, and would reject browser requests from the standalone frontend. Both added.
- **Fragile crop-boundary handling.** The face-detection crop didn't guard against a zero-area box after margin/clipping, which could crash the resize step downstream. Fixed with an explicit degenerate-box check.
- **Unweighted loss on an unbalanced dataset.** The three source datasets are not guaranteed to contribute equal counts per class; training used a plain, unweighted cross-entropy loss. Fixed with inverse-frequency class weighting computed from the actual training split.

## Limitations

- Model performance depends entirely on the training data you supply — this repository does not ship a trained checkpoint or verified accuracy numbers.
- Lighting, image quality, camera angle, and demographic representation in the training data all affect real-world accuracy.
- The model is trained against the specific spoof types present in the source datasets (print/screen replay); unseen spoofing techniques (e.g. 3D masks, deepfake video replay) may not be detected reliably.
- The current pipeline classifies a single face per image (the largest detected face); multiple simultaneous faces are not handled.
- Splitting is done per-image, not per-identity/per-video — if a source dataset contains multiple frames of the same person or video across the split, that could inflate evaluation metrics through data leakage. This is worth checking against the specific dataset versions you use.
- This is a research/portfolio implementation, not a certified or production-grade biometric security system, and should not be treated as a liveness-detection guarantee.

## Future Improvements

- Identity-disjoint (subject-level) train/val/test splitting to eliminate any risk of leakage across near-duplicate frames.
- Threshold calibration and a documented decision policy for borderline confidence scores.
- Broader spoof-attack coverage (video replay, 3D masks) if suitable data becomes available.
- Batch/video inference mode in the API for multi-frame temporal consistency checks.

## License

MIT — see [LICENSE](LICENSE). The source datasets referenced by this project (Face Mask 12K, CelebA-Spoof, Anti-Spoofing) remain subject to their own respective licenses and are not redistributed here.
