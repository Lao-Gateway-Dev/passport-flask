import os 
import cv2
import numpy as np
from flask import Flask, request, jsonify, url_for, send_from_directory
from ultralytics import YOLO
import logging
from functools import wraps
from typing import Tuple, Dict
import time

from process import prepare_image

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join("static", "uploads")
PROCESSED_FOLDER = os.path.join("static", "predicts")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load YOLO model
model = YOLO("weights/passport11x.pt")
CLASS_NAMES = {0: "mrz", 1: "passport"}
CONFIDENCE_THRESHOLD = 0.7

@app.route("/")
def index():
    return jsonify({"status": "healthy"}), 200

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"}), 200

def allowed_file(filename: str) -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/<filename>")
def serve_uploaded_file(filename):
    return send_from_directory("static/uploads", filename)

def error_response(message: str, status_code: int = 400) -> Tuple[Dict, int]:
    return jsonify({
        "success": False,
        "error": message
    }), status_code

def timing_decorator(f):
    """Decorator to measure execution time of functions."""
    @wraps(f)
    def wrap(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        logger.info(f'{f.__name__} took {end-start:.2f} seconds to execute')
        return result
    return wrap

@app.route("/upload", methods=["POST"])
@timing_decorator
def upload_image():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image provided"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No selected file"}), 400
    
    # Save uploaded image
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    # Process image
    result_path, detections = process_image(image_path)
    if result_path is None or detections is None:
        return jsonify({"message": "Passport not detected"}), 200

    image_url = url_for("static", filename=file.filename, _external=True)
    predict_url = url_for("static", filename=f"predicts/{file.filename.rsplit('.', 1)[0]}/{file.filename.rsplit('.', 1)[0]}_result.png", _external=True)
    
    if result_path is None:
        return jsonify(detections), 500

    return jsonify({
        "message": "Image processed successfully",
        "original_image": image_url,
        "predict_image": predict_url,
        "detections": detections
    })

def process_image(image_path):
    processed_path = os.path.join(PROCESSED_FOLDER, os.path.splitext(os.path.basename(image_path))[0])
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Could not read image: {image_path}")
        return None, None
    
    logger.info(f"Processing image: {image_path}")
    processed_image = prepare_image(image_path, processed_path)

    image = cv2.imread(processed_image)
    
    if image is None or not isinstance(image, np.ndarray):
        logger.error(f"Processed image is not a valid NumPy array: {image}")
        return None, None
    
    results = model(image)
    if results is None:
        logger.info('Passport not detected')
        return None, None
    
    detections = extract_detections(results[0])
    if detections is None:
        logger.info(f"No MRZ detected inside a passport")  
        return None, None
    
    # Draw bounding boxes
    for det in detections.get("passports", []):
        draw_bbox(image, det, (0, 255, 0), "Passport")
    for det in detections.get("mrz", []):
        draw_bbox(image, det, (0, 0, 255), "MRZ")
    
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    predict_path = os.path.join(processed_path, f"{base_filename}_result.png")
    try:
        cv2.imwrite(predict_path, image)
        logger.info(f"Predict image saved at {predict_path}")
    except Exception as e:
        logger.error(f"Failed to save predict image: {e}")
        return None, None
    
    return processed_path, detections

def extract_detections(result):
    detections = {"passports": [], "mrz": []}
    
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    
    for i in range(len(class_ids)):
        if confidences[i] >= CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = map(float, boxes[i])
            confidence = float(confidences[i])
            class_id = int(class_ids[i])
            
            entry = {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": confidence
            }
            
            if CLASS_NAMES[class_id] == "passport":
                detections["passports"].append(entry)
            elif CLASS_NAMES[class_id] == "mrz":
                detections["mrz"].append(entry)

    # Filter MRZ: Keep only if inside a passport
    filtered_mrz = []
    for mrz in detections["mrz"]:
        mx1, my1, mx2, my2, m_conf = mrz["x1"], mrz["y1"], mrz["x2"], mrz["y2"], mrz["confidence"]
        for passport in detections["passports"]:
            px1, py1, px2, py2, p_conf = passport["x1"], passport["y1"], passport["x2"], passport["y2"], passport["confidence"]
            if px1 <= mx1 and py1 <= my1 and px2 >= mx2 and py2 >= my2:
                filtered_mrz.append(mrz)
                break  # Ensure one MRZ per passport

    # If no MRZ is inside a passport, skip processing
    if not filtered_mrz:
        return None
    
    return detections

def draw_bbox(image, bbox, color, label):
    if not isinstance(image, np.ndarray):
        logger.error(f"Image is not a valid NumPy array in draw_bbox function.")
        return
    
    x1, y1, x2, y2, conf = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"], bbox["confidence"]
    label_text = f"{label} {conf:.2f}"
    
    h, w = image.shape[:2]
    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
    
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

if __name__ == "__main__":
    app.run(debug=True)
