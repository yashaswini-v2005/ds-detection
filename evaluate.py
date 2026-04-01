import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
MODEL_PATH = "models/downsyndrome_vgg16.h5"
VAL_DIR = "processed_dataset/"  # validation dataset path

# Validation data generator - only rescaling, no augmentation
val_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(128, 128),  # same as training target size
    batch_size=16,
    class_mode="binary",
    subset="validation",
    shuffle=False  # important for consistent class order
)

# Load the trained model
model = load_model(MODEL_PATH)

# Evaluate the model on validation data
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"\nValidation Accuracy: {val_accuracy * 100:.2f}%\n")

# Generate predictions
val_generator.reset()
pred_probs = model.predict(val_generator)
threshold = 0.5
pred_labels = (pred_probs > threshold).astype(int).reshape(-1)

# True labels from generator
true_labels = val_generator.classes

# Classification report
print("Classification Report:")
print(classification_report(true_labels, pred_labels, target_names=['Normal', 'Down Syndrome']))

# Confusion matrix
cm = confusion_matrix(true_labels, pred_labels)

# Plot confusion matrix heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Normal', 'Down Syndrome'],
    yticklabels=['Normal', 'Down Syndrome']
)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix')
plt.show()
