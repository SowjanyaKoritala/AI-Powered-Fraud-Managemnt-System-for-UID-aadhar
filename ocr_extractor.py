import cv2
import pytesseract
import re
import os

# Path to Tesseract executable (for Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def extract_aadhaar_details(image_path):
    img = cv2.imread(image_path)

    # OCR with positional data
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    aadhaar_no = "Not Found"
    name = "Not Found"
    dob = "Not Found"
    gender = "Not Found"

    for i, text in enumerate(data['text']):
        if not text.strip():
            continue

        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        text_upper = text.upper()

        # Aadhaar Number pattern
        if re.match(r'\d{4}\s?\d{4}\s?\d{4}', text_upper):
            aadhaar_no = re.sub(r'\s+', ' ', text)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(img, "Aadhaar", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Date of Birth
        elif re.match(r'\d{2}/\d{2}/\d{4}', text):
            dob = text
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(img, "DOB", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Gender
        elif text_upper in ["MALE", "FEMALE"]:
            gender = text.capitalize()
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(img, "Gender", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Name (above DOB or Gender)
        elif text.isalpha() and len(text) > 3 and name == "Not Found":
            name = text
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(img, "Name", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Save detected image
    output_dir = "static/results"
    os.makedirs(output_dir, exist_ok=True)
    detected_path = os.path.join(output_dir, f"detected_{os.path.basename(image_path)}")
    cv2.imwrite(detected_path, img)

    fraud_score = 0 if aadhaar_no != "Not Found" and dob != "Not Found" else 60
    label = "Real Aadhaar" if fraud_score < 50 else "Fake Aadhaar"

    return {
        "aadhaar_number": aadhaar_no,
        "name": name,
        "dob": dob,
        "gender": gender,
        "fraud_score": fraud_score,
        "label": label,
        "detected_image": detected_path.replace("\\", "/")
    }
