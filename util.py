# util.py
import string
import easyocr
import cv2
import numpy as np
import streamlit as st
import re


@st.cache_resource
def load_ocr_reader():
    st.info("Initializing OCR reader... This may take a moment on the first run.")
    reader = easyocr.Reader(['en'], gpu=False, recog_network='latin_g2')
    st.success("✅ OCR reader ready!")
    return reader


LICENSE_PLATE_CHARS = string.ascii_uppercase + string.digits

CHAR_CORRECTION_MAP = {
    'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8',
    'G': '6', 'A': '4', 'Q': '0',
}


def write_csv(results, output_path):
    # This function remains unchanged
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{},{}\n'.format(
            'frame_nmr', 'filename', 'car_id', 'car_bbox',
            'license_plate_bbox', 'license_plate_bbox_score',
            'license_number', 'license_number_score'
        ))
        # ... (rest of the function is the same)
        f.close()


def get_skew_angle(image) -> float:
    """Finds the skew angle of the text in the image."""
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = contours[0]
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    return angle


def rotate_image(image, angle: float):
    """Rotates an image to correct for skew."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def preprocess_for_ocr(image):
    """Improved preprocessing pipeline for OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale_factor = max(1.5, 300 / gray.shape[1] if gray.shape[1] > 0 else 1.5)
    width = int(gray.shape[1] * scale_factor)
    height = int(gray.shape[0] * scale_factor)
    if width == 0 or height == 0: return gray  # return original gray if resize is not possible
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LANCZOS4)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_contrast = clahe.apply(resized)
    denoised = cv2.bilateralFilter(enhanced_contrast, 9, 75, 75)
    binary_image = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    angle = get_skew_angle(binary_image)
    if -15 < angle < 15 and angle != 0:
        deskewed_image = rotate_image(binary_image, angle)
    else:
        deskewed_image = binary_image
    kernel = np.ones((1, 1), np.uint8)
    final_image = cv2.morphologyEx(deskewed_image, cv2.MORPH_OPEN, kernel)
    return final_image


def clean_plate_text(text):
    """Cleans the OCR'd text."""
    base_cleaned_text = "".join(char for char in text if char in LICENSE_PLATE_CHARS).upper()
    corrected_text = "".join(CHAR_CORRECTION_MAP.get(char, char) for char in base_cleaned_text)
    return corrected_text


def is_valid_license_plate(text):
    """Checks if the text conforms to a common license plate format."""
    if not (4 <= len(text) <= 8):
        return False
    if re.match(r'^[A-Z]{3}[0-9]{4}$', text): return True
    if re.match(r'^[A-Z]{4}[0-9]{3}$', text): return True
    if re.match(r'^[A-Z0-9]{4,8}$', text) and (
            any(c.isalpha() for c in text) and any(c.isdigit() for c in text)): return True
    return False


def read_license_plate(license_plate_crop):
    """Read the license plate text using an improved pipeline."""
    reader = load_ocr_reader()
    processed_plate = preprocess_for_ocr(license_plate_crop)
    detections = reader.readtext(
        processed_plate, allowlist=LICENSE_PLATE_CHARS, paragraph=False,
        decoder='beamsearch', batch_size=4, contrast_ths=0.2, adjust_contrast=0.7
    )
    if not detections:
        return None, None

    full_text = "".join([det[1] for det in detections])
    avg_score = sum([det[2] for det in detections]) / len(detections) if detections else 0
    cleaned_text = clean_plate_text(full_text)

    if is_valid_license_plate(cleaned_text):
        return cleaned_text, avg_score
    else:
        for i in range(len(cleaned_text)):
            for j in range(i + 4, len(cleaned_text) + 1):
                substring = cleaned_text[i:j]
                if is_valid_license_plate(substring):
                    return substring, avg_score
    return None, None


# The get_car function is no longer used by main.py but can be kept
def get_car(license_plate, vehicle_detections):
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