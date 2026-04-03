import cv2
import numpy as np
import pickle

# -------- FEATURE EXTRACTION --------
def extract_features(image):
    image = cv2.resize(image, (128, 128))
    
    # COLOR FEATURES
    hist = cv2.calcHist([image], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    cv2.normalize(hist, hist)
    color_features = hist.flatten()
    
    # EDGE FEATURES
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges = cv2.resize(edges, (32, 32))
    edge_features = edges.flatten()
    
    # COMBINE FEATURES
    features = np.hstack([color_features, edge_features])

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Load test image
image = cv2.imread("test.jpg")

features = extract_features(image)

prediction = model.predict([features])

print("Predicted waste type:", prediction[0])