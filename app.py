import os
import cv2
import numpy as np
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

model = joblib.load("model.pkl")

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/register")
def register():
    return render_template("register.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/", methods=["POST"])
def predict():
    file = request.files["file"]

    if file:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        img = cv2.resize(img, (32, 32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        features = img.flatten().reshape(1, -1)

        prediction = model.predict(features)[0]

        return render_template("index.html", prediction=prediction)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)