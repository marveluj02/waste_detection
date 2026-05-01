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
    "username": "wastemonger",
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

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/contact', methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        return render_template("contact.html", success=True)

    return render_template("contact.html", success=False)

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


@app.route('/contact', methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        message = request.form.get("message")

        print(name, email, message)

        return redirect(url_for("dashboard"))

    return render_template("contact.html")

@app.route('/pickup', methods=["GET", "POST"])
def pickup():
    if request.method == "POST":
        waste_type = request.form.get("waste_type")
        date = request.form.get("date")
        time = request.form.get("time")
        address = request.form.get("address")

        print(waste_type, date, time, address)  # check in terminal

        return render_template("pickup.html", success=True)

    return render_template("pickup.html", success=False)
# ----------------- LOGOUT -----------------

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

@app.route('/pickup', methods=["GET", "POST"])
def pickup():
    if request.method == "POST":
        waste_type = request.form.get("waste_type")
        date = request.form.get("date")
        time = request.form.get("time")
        address = request.form.get("address")

        print(waste_type, date, time, address)  # for now

        return redirect(url_for("dashboard"))

    return render_template("pickup.html")

# ----------------- RUN APP -----------------

if __name__ == "__main__":
    app.run(debug=True)