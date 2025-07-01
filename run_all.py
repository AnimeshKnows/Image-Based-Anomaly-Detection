import os
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import cv2
import numpy as np
from train import train_model
from inference import run_inference

# === CONFIG ===
TRAIN_DIR = r"H:\image-anomaly-detection\bottle\train\good"
TEST_DIR = r"H:\image-anomaly-detection\bottle\test"
OUTPUT_DIR = r"H:\image-anomaly-detection\src\outputs"
MODEL_PATH = r"H:\image-anomaly-detection\src\models\cae_bottle.h5"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20

# === Step 1: Train the model ===
model, history = train_model(TRAIN_DIR, IMG_SIZE, BATCH_SIZE, EPOCHS, MODEL_PATH)

# === Step 2: Plot training loss ===
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CAE Training Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss.png")
# ✅ Show live plot without blocking script
plt.show()

# === Step 3: Run inference & save results ===
print("[INFO] Running inference...")
results = run_inference(model, TEST_DIR, OUTPUT_DIR, IMG_SIZE, samples_per_class=3)

# === Step 4: Plot and save 2x2 grids for each result ===
print("[INFO] Creating visual result grids...")

visuals_dir = os.path.join(OUTPUT_DIR, "visuals")
os.makedirs(visuals_dir, exist_ok=True)

for item in results:
    input_img = item['input']
    recon_img = item['recon']
    heatmap = item['heatmap']
    overlay = item['overlay']
    defect_type = item['class']
    name = item['name']

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    axs[0, 0].imshow(input_img)
    axs[0, 0].set_title("Original")
    axs[0, 1].imshow(recon_img)
    axs[0, 1].set_title("Reconstruction")
    axs[1, 0].imshow(heatmap, cmap='hot')
    axs[1, 0].set_title("Anomaly Heatmap")
    axs[1, 1].imshow(overlay)
    axs[1, 1].set_title("Overlay")

    for ax in axs.ravel():
        ax.axis('off')

    plt.suptitle(f"{defect_type.upper()} - {name}", fontsize=14)
    plt.tight_layout()

    # Save to visuals subfolder
    save_path = os.path.join(visuals_dir, f"{defect_type}_{name}_grid.png")
    plt.savefig(save_path)
    plt.close()

    print("[INFO] Done. All visualizations saved in:", visuals_dir)

    
    # Save without showing
    save_path = os.path.join(OUTPUT_DIR, defect_type, f"{name}_grid.png")
    plt.savefig(save_path)
    plt.close()

print("[INFO] Done. All results and plots saved.")