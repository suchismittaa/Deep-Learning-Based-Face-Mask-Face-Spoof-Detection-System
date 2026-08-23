"""FastAPI inference service for the Real / Spoof / Masked face classifier.

Run:
    uvicorn app:app --reload
"""
import io
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import FaceSpoofMaskClassifier
from src.face_detection import FaceDetector

app = FastAPI(
    title="Face Mask & Spoof Detection API",
    description="ResNet18 + MTCNN inference API for real / spoof / masked face classification.",
    version="1.0.0",
)

# The bundled frontend is a static file opened directly in the browser
# (file:// or any static host), so we allow any origin rather than trying
# to guess where it will be served from. There are no cookies/auth here,
# so allow_credentials stays False.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best_model.pth")

model = None
class_names = None
model_error = None

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "class_to_idx" in checkpoint:
        # Current checkpoint format (see train.py): label order comes from
        # the checkpoint itself, so it can never drift out of sync with
        # how the model was actually trained.
        class_to_idx = checkpoint["class_to_idx"]
        class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
        state_dict = checkpoint["model_state_dict"]
    else:
        # Backwards-compatible fallback for a raw state_dict checkpoint.
        # ImageFolder sorts class folders alphabetically, so this is the
        # correct default order for this project's three classes.
        class_names = ["masked", "real", "spoof"]
        state_dict = checkpoint

    model = FaceSpoofMaskClassifier(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
except FileNotFoundError:
    model_error = f"Model weights not found at {MODEL_PATH}. Train the model first (see README)."
except Exception as exc:  # noqa: BLE001 - surfaced via /health and /predict
    model_error = f"Model could not be loaded: {exc}"

face_detector = FaceDetector()
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@app.get("/")
def root():
    return {"service": "Face Mask & Spoof Detection API", "status": "ok", "model_loaded": model is not None}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(DEVICE),
        "classes": class_names,
        "error": model_error,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail=model_error or "Model is not loaded.")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    try:
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="Uploaded file is not a readable image.")

        # PIL already decodes to RGB - no BGR<->RGB conversion needed here.
        image_rgb = np.array(image)

        cropped_face = face_detector.detect_and_crop(image_rgb)
        if cropped_face is None:
            return JSONResponse(status_code=400, content={"message": "No face detected in the uploaded image."})

        face_tensor = transform(cropped_face).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probabilities = F.softmax(model(face_tensor), dim=1)[0]
            confidence, predicted = torch.max(probabilities, 0)

        return {
            "prediction": class_names[predicted.item()],
            "confidence": round(float(confidence), 4),
            "all_probabilities": {
                class_names[i]: round(float(probabilities[i]), 4) for i in range(len(class_names))
            },
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")
