import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from cae import build_cae  # import your model architecture

# === Config ===
MODEL_PATH = "models/cae_bottle.h5"
TEST_DIR = r"H:\image-anomaly-detection\bottle\test"
OUTPUT_DIR = "outputs"
IMG_SIZE = (128, 128)

# === Load and Preprocess ===
def load_and_preprocess(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype('float32') / 255.0
    return img

# === Load Model ===
print("[INFO] Loading model...")
model = build_cae(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
model.load_weights(MODEL_PATH)
print("[INFO] Model loaded.")

# === Inference Loop ===
print("[INFO] Running inference...")
for defect_type in os.listdir(TEST_DIR):
    class_dir = os.path.join(TEST_DIR, defect_type)
    output_class_dir = os.path.join(OUTPUT_DIR, defect_type)
    os.makedirs(output_class_dir, exist_ok=True)

    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        img = load_and_preprocess(img_path)
        if img is None:
            continue

        input_img = np.expand_dims(img, axis=0)  # Add batch dim
        recon_img = model.predict(input_img)[0]

        # === Anomaly Map (Absolute Error) ===
        error_map = np.abs(img - recon_img)
        heatmap = np.mean(error_map, axis=-1)  # Grayscale

        # === Normalize heatmap for display ===
        heatmap_norm = cv2.normalize(heatmap.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
        heatmap_colored = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)

        # === Overlay on original ===
        overlay = cv2.addWeighted((img * 255).astype(np.uint8), 0.6, heatmap_colored, 0.4, 0)

        # === Save Results ===
        cv2.imwrite(os.path.join(output_class_dir, f"{os.path.splitext(img_name)[0]}_input.png"), (img * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(output_class_dir, f"{os.path.splitext(img_name)[0]}_recon.png"), (recon_img * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(output_class_dir, f"{os.path.splitext(img_name)[0]}_heatmap.png"), heatmap_colored)
        cv2.imwrite(os.path.join(output_class_dir, f"{os.path.splitext(img_name)[0]}_overlay.png"), overlay)

print("[INFO] Inference completed. Results saved in:", OUTPUT_DIR)
