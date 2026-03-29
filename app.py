from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
from flask_cors import CORS
import cv2
import numpy as np
import os

app = Flask(__name__, static_folder=".")
CORS(app)

# Load your model
model = YOLO("assets\\ai model\\best.pt")

# Route to serve HTML pages
@app.route("/")
def index():
    return send_from_directory(".", "inspection.html")  # default page

@app.route("/<path:path>")
def serve_file(path):
    if os.path.exists(path):
        return send_from_directory(".", path)
    else:
        return "File not found", 404

# Your prediction API
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    results = model(img)

    detections = []
    for r in results:
        for b in r.boxes:
            detections.append({
                "class": int(b.cls[0]),
                "confidence": float(b.conf[0]),
                "bbox": b.xyxy[0].tolist()
            })

    return jsonify(detections)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)