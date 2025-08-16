import cv2
import os
import numpy as np
from tqdm import tqdm

# Paths
DATASET_DIR = "dataset/train"
CATEGORIES = ["cat", "dog"]
IMG_SIZE = 64

features = []
labels = []

print("[INFO] Loading and processing images...")

for category in CATEGORIES:
    path = os.path.join(DATASET_DIR, category)
    label = 0 if category == "cat" else 1
    
    for img_name in tqdm(os.listdir(path)):
        try:
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # grayscale
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))       # resize
            features.append(img.flatten())                    # flatten
            labels.append(label)
        except Exception as e:
            continue

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

print(f"[INFO] Total samples: {len(X)}")

# Save processed data
np.savez_compressed("cat_dog_features.npz", X=X, y=y)
print("[INFO] Preprocessing complete. Saved as cat_dog_features.npz")
