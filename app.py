import os
import cv2
import numpy as np
import joblib
from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = "secret123"

# ---------------- MODEL ----------------
MODEL_PATH = "model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None
    print("❌ Model file not found")

# ---------------- UPLOAD FOLDER ----------------
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ---------------- USER ----------------
USER = {
    "username": "wastemonger",
    "password": "1234"
}

# ---------------- LOGIN ----------------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == USER["username"] and password == USER["password"]:
            session["user"] = username
            return redirect(url_for("home"))
        else:
            return render_template("login.html", error="Invalid login")

    return render_template("login.html")

# ---------------- HOME ----------------
@app.route("/home")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("home.html")

# ---------------- DASHBOARD ----------------
@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    prediction = None
    image_path = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            if model is not None:
                img = cv2.imread(filepath)
                img = cv2.resize(img, (32, 32))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                features = img.flatten().reshape(1, -1)
                prediction = model.predict(features)[0]
            else:
                prediction = "Model not loaded"

            image_path = filepath

    return render_template("dashboard.html",
                           prediction=prediction,
                           image_path=image_path)

# ---------------- ABOUT ----------------
@app.route("/about")
def about():
    return render_template("about.html")

# ---------------- CONTACT ----------------
@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        return render_template("contact.html", success=True)

    return render_template("contact.html", success=False)

# ---------------- PICKUP ----------------
@app.route("/pickup", methods=["GET", "POST"])
def pickup():
    if request.method == "POST":
        return render_template("pickup.html", success=True)

    return render_template("pickup.html", success=False)

# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)