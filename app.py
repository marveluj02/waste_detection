import os
import cv2
import numpy as np
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)
model = joblib.load("model.pkl")

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            img = cv2.imread(filepath)

            img = cv2.resize(img, (32, 32))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            features = img.flatten().reshape(1, -1)

            probs = model.predict_proba(features)[0]
            confidence = round(max(probs) * 100, 2)
            prediction = model.classes_[np.argmax(probs)]

            if confidence < 50:
                prediction = "Uncertain"

            image_path = filepath

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)