import os
import sys
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from torchvision import transforms

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import FaceSpoofMaskClassifier
from src.face_detection import FaceDetector

# --- INITIALIZATION ---
app = FastAPI(title="Face Mask & Spoof Detection API")

# Load model and class names
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best_model.pth')

try:
    model = FaceSpoofMaskClassifier(num_classes=3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model not found at {MODEL_PATH}. Please train the model first.")
    model = None

# Initialize face detector and transformations
face_detector = FaceDetector()
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
CLASS_LABELS = ["real", "spoof", "masked"]

# --- API ENDPOINTS ---
@app.get("/")
async def root():
    return {"message": "API is running. Go to /docs to test."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded is not an image.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = np.array(image)
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        cropped_face = face_detector.detect_and_crop(image_rgb)
        
        if cropped_face is None:
            return JSONResponse(status_code=400, content={"message": "No face detected."})
        
        face_tensor = transform(cropped_face).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(face_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            predicted_class = CLASS_LABELS[predicted.item()]
            confidence_score = confidence.item()
        
        return {
            "prediction": predicted_class,
            "confidence": round(confidence_score, 4),
            "all_probabilities": {
                CLASS_LABELS[i]: round(prob.item(), 4) for i, prob in enumerate(probabilities[0])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")