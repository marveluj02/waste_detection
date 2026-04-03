import cv2
import joblib
import numpy as np

# Load trained model
model = joblib.load("model.pkl")

cap = cv2.VideoCapture(0)
print("[INFO] Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --------- CENTER CROP (reduces background noise) ----------
    h, w, _ = frame.shape
    crop = frame[h//4:h*3//4, w//4:w*3//4]

    # --------- SAME PREPROCESSING AS TRAINING ----------
    img = cv2.resize(crop, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    features = img.flatten().reshape(1, -1)

    # --------- PREDICTION WITH CONFIDENCE ----------
    probs = model.predict_proba(features)[0]
    confidence = max(probs)
    prediction = model.classes_[np.argmax(probs)]

    # If confidence is low, show "Uncertain"
    if confidence < 0.5:
        prediction = "Uncertain"

    # --------- DISPLAY RESULT ----------
    text = f"{prediction} ({confidence*100:.1f}%)"

    cv2.putText(frame, text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("Waste Detection", frame)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()