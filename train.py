import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

DATASET_PATH = "dataset"
labels = ["plastic", "metal", "paper", "organic", "glass"]

data = []
target = []

print("[INFO] Loading images...")

for label in labels:
    folder = os.path.join(DATASET_PATH, label)

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (32, 32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        features = img.flatten()

        data.append(features)
        target.append(label)

print("[INFO] Preparing data...")
data = np.array(data)
target = np.array(target)

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, stratify=target, random_state=42
)

print("[INFO] Training model...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

print(f"[RESULT] Accuracy: {accuracy:.2f}")

joblib.dump(model, "model.pkl")
print("[INFO] Model saved as model.pkl")