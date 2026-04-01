# predict.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
MODEL_PATH = "models/downsyndrome_vgg16.h5"
model = load_model(MODEL_PATH)

IMG_SIZE = 128  # must match training size

def preprocess_image(img_path):
    """Reads and preprocesses image for prediction."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"❌ Image not found: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # Convert to RGB
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize
    img = img.astype("float32") / 255.0          # Normalize
    img = np.expand_dims(img, axis=0)            # Add batch dim
    return img

def predict(img_path):
    """Predicts probability and returns detailed result."""
    img = preprocess_image(img_path)
    prob = model.predict(img, verbose=0)[0][0]   # Probability for Down class

    down_prob = round(prob * 100, 2)
    normal_prob = round((1 - prob) * 100, 2)

    print(f"🔍 Raw Model Probability (Down syndrome): {down_prob}%")
    print(f"🔍 Raw Model Probability (Normal): {normal_prob}%")

    if prob > 0.5:
        result = (
            f"Prediction: {down_prob}% chance of Down syndrome features detected.\n"
            "⚠️ Suggestion: Please consult a pediatrician for medical confirmation."
        )
    else:
        result = (
            f"Prediction: {normal_prob}% chance of being normal.\n"
            "✅ Suggestion: No major indicators found, but consult a pediatrician if concerned."
        )

    return result


# Example test run
if __name__ == "__main__":
    test_image = "assets/test_baby3.jpg"   # update with your image path
    print(predict(test_image))
