import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

# Paths
train_dir = "processed_dataset/"
model_path = "models/downsyndrome_vgg16.h5"

# ✅ Data Augmentation (creates more training images artificially)
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2   # 80% training, 20% validation
)

# Train/Validation split
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=16,
    class_mode="binary",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=16,
    class_mode="binary",
    subset="validation"
)

# ✅ Load Pretrained VGG16 (without top layers)
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(128, 128, 3))

# Freeze base layers (don’t retrain ImageNet weights)
for layer in base_model.layers:
    layer.trainable = False

# ✅ Add custom classifier on top
x = Flatten()(base_model.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

# ✅ Train model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    verbose=1
)

# ✅ Save model
os.makedirs("models", exist_ok=True)
model.save(model_path)
print(f"🎉 VGG16 Transfer Learning Model saved at {model_path}")
