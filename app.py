import os
import cv2
import re
import pandas as pd
import pytesseract
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime
import numpy as np

# ------------------- CONFIG -------------------
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
CSV_PATH = os.path.join(RESULTS_FOLDER, "extracted_details.csv")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ------------------- OCR + BOXES -------------------
def preprocess_image(image):
    """Preprocess image for better OCR accuracy"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
    return gray

def extract_aadhaar_details(image):
    """Extract Aadhaar card details using OCR"""
    gray = preprocess_image(image)
    text = pytesseract.image_to_string(gray)
    text_upper = text.upper().replace("\n", " ")

    # Aadhaar number
    aadhaar_match = re.search(r"\b\d{4}\s\d{4}\s\d{4}\b", text_upper)
    aadhaar_number = aadhaar_match.group() if aadhaar_match else None

    # DOB
    dob_match = re.search(r"\b\d{2}/\d{2}/\d{4}\b", text_upper)
    dob = dob_match.group() if dob_match else None

    # Gender
    if "MALE" in text_upper:
        gender = "Male"
    elif "FEMALE" in text_upper:
        gender = "Female"
    else:
        gender = None

    # Name
    name_match = re.findall(r"[A-Z][A-Z]+\s[A-Z][A-Z]+", text_upper)
    name = name_match[0] if name_match else None

    return {
        "aadhaar_number": aadhaar_number,
        "name": name,
        "dob": dob,
        "gender": gender,
        "raw_text": text.strip()
    }

def draw_highlighted_boxes(image):
    """Draw bounding boxes around detected text"""
    boxes = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    boxed_image = image.copy()
    n_boxes = len(boxes['level'])
    for i in range(n_boxes):
        if int(boxes['conf'][i]) > 40:  # only confident detections
            (x, y, w, h) = (boxes['left'][i], boxes['top'][i],
                            boxes['width'][i], boxes['height'][i])
            cv2.rectangle(boxed_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return boxed_image

def is_aadhaar_card(text):
    """Basic Aadhaar validity check"""
    if text is None:
        return False
    return bool(re.search(r"\b\d{4}\s\d{4}\s\d{4}\b", text)) or "AADHAAR" in text.upper()

def check_fraud(details):
    """Simple heuristic to detect fake Aadhaar"""
    score = 0
    if not details.get("aadhaar_number"):
        score += 1
    if not details.get("gender"):
        score += 0.3
    name = details.get("name", "")
    if len(name or "") < 3 or not any(c.isalpha() for c in name or ""):
        score += 0.3
    return min(score, 1)

# ------------------- ROUTES -------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty file uploaded"}), 400

    filename = secure_filename(file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)

    image = cv2.imread(image_path)
    if image is None:
        return jsonify({"error": "Failed to read uploaded image"}), 400

    details = extract_aadhaar_details(image)
    valid_aadhaar = is_aadhaar_card(details["raw_text"])
    fraud_score = check_fraud(details)
    label = "Fake Aadhaar" if fraud_score >= 0.5 else "Valid Aadhaar"

    # Draw boxes and save detected image
    boxed_image = draw_highlighted_boxes(image)
    boxed_filename = f"boxed_{filename}"
    boxed_path = os.path.join(RESULTS_FOLDER, boxed_filename)
    cv2.imwrite(boxed_path, boxed_image)

    record = {
        "filename": filename,
        "aadhaar_number": details.get("aadhaar_number") or "Not Found",
        "name": details.get("name") or "Not Found",
        "dob": details.get("dob") or "Not Found",
        "gender": details.get("gender") or "Not Found",
        "fraud_score": fraud_score,
        "label": label,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])
    df.to_csv(CSV_PATH, index=False)

    return jsonify({
        "original_image": f"/{UPLOAD_FOLDER}/{filename}",
        "boxed_image": f"/{RESULTS_FOLDER}/{boxed_filename}",
        "ocr_details": record,
        "prediction": label
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
