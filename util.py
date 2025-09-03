

import string
import easyocr
import cv2
import numpy as np
import streamlit as st  # <-- ADD THIS IMPORT

# --- REMOVE THE OLD LINE ---
# reader = easyocr.Reader(['en'], gpu=False)  <-- DELETE THIS

# --- ADD THIS NEW CACHED FUNCTION ---
@st.cache_resource
def load_ocr_reader():
    """Loads the EasyOCR reader into cache."""
    st.info("Initializing OCR reader... This may take a moment on the first run.")
    reader = easyocr.Reader(['en'], gpu=False)
    st.success("✅ OCR reader ready!")
    return reader

def write_csv(results, output_path):
    """
    Write the results to a CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{},{}\n'.format(
            'frame_nmr', 'filename', 'car_id', 'car_bbox',
            'license_plate_bbox', 'license_plate_bbox_score',
            'license_number', 'license_number_score'
        ))

        for frame_nmr in results.keys():
            filename = results[frame_nmr].get('filename', '')
            for car_id in results[frame_nmr].keys():
                if car_id == 'filename':
                    continue

                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():

                    f.write('{},{},{},{},{},{},{},{}\n'.format(
                        frame_nmr,
                        filename,
                        car_id,
                        '[{} {} {} {}]'.format(
                            results[frame_nmr][car_id]['car']['bbox'][0],
                            results[frame_nmr][car_id]['car']['bbox'][1],
                            results[frame_nmr][car_id]['car']['bbox'][2],
                            results[frame_nmr][car_id]['car']['bbox'][3]
                        ),
                        '[{} {} {} {}]'.format(
                            results[frame_nmr][car_id]['license_plate']['bbox'][0],
                            results[frame_nmr][car_id]['license_plate']['bbox'][1],
                            results[frame_nmr][car_id]['license_plate']['bbox'][2],
                            results[frame_nmr][car_id]['license_plate']['bbox'][3]
                        ),
                        results[frame_nmr][car_id]['license_plate']['bbox_score'],
                        results[frame_nmr][car_id]['license_plate']['text'],
                        results[frame_nmr][car_id]['license_plate']['text_score']
                    ))
        f.close()


def preprocess_for_ocr(image):
    """
    Applies a series of preprocessing steps to an image to improve OCR accuracy.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    scale_factor = 100 / blurred.shape[0]
    width = int(blurred.shape[1] * scale_factor)
    height = int(blurred.shape[0] * scale_factor)
    resized = cv2.resize(blurred, (width, height), interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_contrast = clahe.apply(resized)
    _, binary_image = cv2.threshold(enhanced_contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image


def clean_plate_text(text):
    """
    Cleans the license plate text by removing non-alphanumeric characters and converting to uppercase.
    """
    return "".join(char for char in text if char.isalnum()).upper()


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.
    Applies preprocessing before sending the image to the OCR reader.
    """
    # --- UPDATE THIS PART ---
    # Get the cached reader instead of using a global variable
    reader = load_ocr_reader()
    # --- END OF UPDATE ---

    processed_plate = preprocess_for_ocr(license_plate_crop)
    detections = reader.readtext(processed_plate)

    if not detections:
        return None, None

    full_text = ""
    total_score = 0
    for bbox, text, score in detections:
        full_text += text
        total_score += score

    avg_score = total_score / len(detections)
    cleaned_text = clean_plate_text(full_text)

    if len(cleaned_text) >= 3:
        return cleaned_text, avg_score
    else:
        return None, None


def get_car(license_plate, vehicle_detections):
    """
    Retrieve the vehicle coordinates based on the license plate coordinates.
    """
    x1, y1, x2, y2, _, _ = license_plate

    for detection in vehicle_detections:
        if len(detection) >= 6:
            xcar1, ycar1, xcar2, ycar2, score, _ = detection
        elif len(detection) == 5:
            xcar1, ycar1, xcar2, ycar2, score = detection
        else:
            continue

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return xcar1, ycar1, xcar2, ycar2, score

    return -1, -1, -1, -1, -1