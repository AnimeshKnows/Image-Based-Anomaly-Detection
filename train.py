import os
import cv2
import numpy as np
import tensorflow as tf
from cae import build_cae
from sklearn.model_selection import train_test_split

# === Config ===
DATA_DIR = r"H:\image-anomaly-detection\bottle\train\good"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20
MODEL_SAVE_PATH = "models/cae_bottle.h5"

# === Load & Preprocess Images ===
def load_images(data_path, img_size):
    images = []
    for img_name in os.listdir(data_path):
        img_path = os.path.join(data_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Use IMREAD_GRAYSCALE if needed
        if img is None:
            continue
        img = cv2.resize(img, img_size)
        img = img.astype('float32') / 255.0
        images.append(img)
    images = np.array(images)
    return images

print("Loading training images...")
X = load_images(DATA_DIR, IMG_SIZE)
print("Total images:", X.shape)

# Add channel dimension if needed
if len(X.shape) == 3:
    X = np.expand_dims(X, axis=-1)  # Grayscale
input_shape = X.shape[1:]

# Split into training and validation sets
X_train, X_val = train_test_split(X, test_size=0.1, random_state=42)

# === Build & Train Model ===
model = build_cae(input_shape)
model.summary()

print("Training model...")
model.fit(X_train, X_train,
          validation_data=(X_val, X_val),
          epochs=EPOCHS,
          batch_size=BATCH_SIZE)

# === Save Model ===
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
