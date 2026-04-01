# ============================================================
#  predict_fixed.py  (Stable version – Full CNN rebuild + SavedModel export)
# ============================================================

import os
import warnings
import cv2
import numpy as np
import mediapipe as mp
import json
import logging
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# ------------------------------------------------------------
# Setup
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Model configuration
MODEL_PATH = os.path.join("models", "downsyndrome_cnn.h5")
SAVEDMODEL_PATH = os.path.join("models", "downsyndrome_savedmodel")
IMG_SIZE = 112  # ✅ Correct input size from your trained model

mp_face = mp.solutions.face_detection


# ============================================================
#  Model Initialization
# ============================================================
def initialize_model():
    """Rebuild trained CNN and save as full model for future reuse"""
    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Model file not found at: {os.path.abspath(MODEL_PATH)}")
        sys.exit(1)

    try:
        # If already saved in SavedModel format, load directly
        if os.path.exists(SAVEDMODEL_PATH):
            model = tf.keras.models.load_model(SAVEDMODEL_PATH)
            logger.info(f"✅ Loaded existing TensorFlow SavedModel from {SAVEDMODEL_PATH}")
            return model

        # Otherwise, rebuild the CNN architecture and load weights
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', padding='same',
                   input_shape=(IMG_SIZE, IMG_SIZE, 3)),  # ✅ Correct shape
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        model.load_weights(MODEL_PATH)
        logger.info(f"✅ Model weights loaded successfully from {MODEL_PATH}")

        # Save as a permanent TensorFlow SavedModel for future use
        model.save(SAVEDMODEL_PATH)
        logger.info(f"💾 Full model saved as TensorFlow SavedModel at {SAVEDMODEL_PATH}")

        model.summary()
        return model

    except Exception as e:
        logger.error(f"❌ Failed to rebuild and load weights: {e}")
        sys.exit(1)


# Initialize model on import
model = initialize_model()


# ============================================================
#  Class Mapping
# ============================================================
mapping_file = os.path.join("models", "class_indices.json")
class_order = None
if os.path.exists(mapping_file):
    try:
        saved = json.load(open(mapping_file))
        inv = {v: k for k, v in saved.items()}
        class_order = [inv[i] for i in sorted(inv.keys())]
    except Exception:
        pass

if class_order is None:
    class_order = ["Down", "Normal"]

down_index = next((i for i, name in enumerate(class_order)
                   if "down" in name.lower()), 0)


# ============================================================
#  Image Validation
# ============================================================
def validate_input_image(img_path):
    """Validate input using Mediapipe face detection"""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return "noface", "❌ Could not read image."

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.15) as detector:
        results = detector.process(img_rgb)
        if not results.detections or len(results.detections) == 0:
            return "noface", "❌ No face detected. Please upload a clear infant image."

    return "valid", ""


# ============================================================
#  Face Cropping
# ============================================================
def crop_face_mediapipe(img_bgr, margin=30, min_conf=0.15):
    """Crop face region using Mediapipe"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]

    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=min_conf) as detector:
        results = detector.process(img_rgb)
        if results.detections and len(results.detections) > 0:
            d = results.detections[0].location_data.relative_bounding_box
            x1 = max(0, int(d.xmin * w) - margin)
            y1 = max(0, int(d.ymin * h) - margin)
            x2 = min(w, int((d.xmin + d.width) * w) + margin)
            y2 = min(h, int((d.ymin + d.height) * h) + margin)
            face = img_rgb[y1:y2, x1:x2]
            if face.size > 0:
                return face
    return None


# ============================================================
#  Preprocessing for Prediction
# ============================================================
def preprocess_for_model(img_path):
    """Load and preprocess image for CNN input"""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"❌ Could not read image: {img_path}")

    face = crop_face_mediapipe(img_bgr)
    if face is None:
        raise ValueError("❌ No face detected for prediction.")

    face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face_norm = face_resized.astype("float32") / 255.0
    return np.expand_dims(face_norm, axis=0)


# ============================================================
#  Prediction Function for Streamlit
# ============================================================
def predict_image_streamlit(img_path, thresh=0.5):
    """Predict Down Syndrome probability for given image"""
    status, msg = validate_input_image(img_path)

    if status != "valid":
        return {"error": msg, "status": "error"}

    try:
        x = preprocess_for_model(img_path)
        pred = float(model.predict(x, verbose=0)[0][0])

        down_prob = pred if down_index == 1 else 1.0 - pred
        down_pct = round(down_prob * 100, 2)
        normal_prob = round(100 - down_pct, 2)

        label = "Down Syndrome Detected" if down_prob >= thresh else "Normal"
        message = ("⚠️ High likelihood of Down Syndrome features detected."
                   if down_prob >= thresh else
                   "✅ Likely normal facial features detected.")

        return {
            "down_prob": down_pct,
            "normal_prob": normal_prob,
            "label": label,
            "message": message,
            "status": "ok"
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {"error": str(e), "status": "error"}
