import os
import cv2
import numpy as np
import joblib

from flask import Flask, render_template, request, redirect, url_for, session

# ----------------- APP SETUP -----------------
app = Flask(__name__)
app.secret_key = "secret123"

# Load trained model
model = joblib.load("model.pkl")

# Upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ----------------- LOGIN SYSTEM -----------------

# Temporary user (you can improve later)
USER = {
    "username": "admin",
    "password": "1234"
}

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == USER["username"] and password == USER["password"]:
            session["user"] = username
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="Invalid login details")

    return render_template("login.html")


# ----------------- DASHBOARD -----------------

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

            # ----------- IMAGE PROCESSING (same as training) -----------
            img = cv2.imread(filepath)
            img = cv2.resize(img, (32, 32))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            features = img.flatten().reshape(1, -1)

            prediction = model.predict(features)[0]

            image_path = filepath

    return render_template(
        "dashboard.html",
        prediction=prediction,
        image_path=image_path
    )


# ----------------- OTHER PAGES -----------------

@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


# ----------------- LOGOUT -----------------

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


# ----------------- RUN APP -----------------

if __name__ == "__main__":
    app.run(debug=True)