import os
import cv2
import mediapipe as mp
import numpy as np

# Input and output folders
INPUT_DIR = "dataset"
OUTPUT_DIR = "processed_dataset"

# Create output folders
os.makedirs(os.path.join(OUTPUT_DIR, "Normal"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "Down"), exist_ok=True)

# Mediapipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def preprocess_and_save(class_name):
    input_path = os.path.join(INPUT_DIR, class_name)
    output_path = os.path.join(OUTPUT_DIR, class_name)

    # Loop through each image
    for i, img_name in enumerate(os.listdir(input_path)):
        img_path = os.path.join(input_path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"⚠️ Skipping {img_name} (not readable)")
            continue

        # Convert to RGB for Mediapipe
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(rgb)

            if not results.detections:
                print(f"❌ No face detected in {img_name}, skipping...")
                continue

            # Take first detected face
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # Add margin
            margin = 20
            x1, y1 = max(0, x - margin), max(0, y - margin)
            x2, y2 = min(iw, x + w + margin), min(ih, y + h + margin)

            # Crop + resize
            face = image[y1:y2, x1:x2]
            face_resized = cv2.resize(face, (128, 128))

            # Save
            save_path = os.path.join(output_path, f"{class_name}_{i+1}.jpg")
            cv2.imwrite(save_path, face_resized)
            print(f"✅ Saved {save_path}")

# Run for both classes
preprocess_and_save("Normal")
preprocess_and_save("Down")

print("🎉 Preprocessing complete! All faces saved to processed_dataset/")
