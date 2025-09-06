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
    reader = easyocr.Reader(['en'], gpu=False)
    st.success("✅ OCR reader ready!")
    return reader


LICENSE_PLATE_CHARS = string.ascii_uppercase + string.digits

# Expanded correction map for more common OCR errors
CHAR_CORRECTION_MAP = {
    'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8',
    'G': '6', 'A': '4', 'Q': '0', 'D': '0', 'T': '7', 'L': '1',
}


def write_csv(results, output_path):
    # This function remains unchanged
    pass  # Keeping it short for brevity


def unsharp_mask(image, amount=2.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def preprocess_for_ocr(image):
    """Improved preprocessing pipeline with sharpening for better OCR."""
    # Ensure image is not empty
    if image is None or image.size == 0:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize for consistency
    scale_factor = max(1.5, 400 / gray.shape[1] if gray.shape[1] > 0 else 1.5)
    width = int(gray.shape[1] * scale_factor)
    height = int(gray.shape[0] * scale_factor)
    if width == 0 or height == 0: return None
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LANCZOS4)

    # Use sharpening to enhance edges
    sharpened = unsharp_mask(resized, amount=2.0)

    # Use Otsu's binarization method, which is great for bimodal images (text vs background)
    _, binary_image = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return binary_image


def clean_plate_text(text):
    """Cleans the OCR'd text."""
    base_cleaned_text = "".join(char for char in text if char in LICENSE_PLATE_CHARS or char == ' ').upper()
    corrected_text = "".join(CHAR_CORRECTION_MAP.get(char, char) for char in base_cleaned_text)
    return corrected_text.replace(' ', '')  # Remove all spaces


def is_valid_license_plate(text):
    """Flexible validation for different license plate formats."""
    if not (5 <= len(text) <= 12):
        return False
    # UK Format (e.g., SN66XMZ)
    if re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{3}$', text): return True
    # Malaysian/Singaporean/Vanity (e.g., PATRIOT4915)
    if re.match(r'^[A-Z]{1,10}[0-9]{1,5}$', text): return True
    # Common US/EU style (e.g., ABC1234)
    if re.match(r'^[A-Z]{3,4}[0-9]{3,4}$', text): return True
    # General Fallback - Requires both letters and numbers
    if re.match(r'^[A-Z0-9]{5,10}$', text) and any(c.isalpha() for c in text) and any(
        c.isdigit() for c in text): return True
    return False


def read_license_plate(license_plate_crop):
    """
    Read the license plate text with sorting, sharpening, and robust validation.
    """
    reader = load_ocr_reader()

    processed_plate = preprocess_for_ocr(license_plate_crop)
    if processed_plate is None:
        return None, None

    detections = reader.readtext(
        processed_plate,
        allowlist=LICENSE_PLATE_CHARS,
        paragraph=False
    )

    if not detections:
        return None, None

    # --- CRITICAL FIX: Sort detections by their horizontal position ---
    detections.sort(key=lambda x: x[0][0][0])

    # Combine sorted detections
    full_text = "".join([det[1] for det in detections])
    avg_score = sum([det[2] for det in detections]) / len(detections) if detections else 0

    cleaned_text = clean_plate_text(full_text)

    if is_valid_license_plate(cleaned_text):
        return cleaned_text, avg_score
    else:
        # Check for valid substrings
        for i in range(len(cleaned_text)):
            for j in range(i + 5, len(cleaned_text) + 1):
                substring = cleaned_text[i:j]
                if is_valid_license_plate(substring):
                    return substring, avg_score

    # If no strict pattern matches, but we have text, return the cleaned version as a last resort
    if len(cleaned_text) >= 5:
        return cleaned_text, avg_score

    return None, None


# The get_car function is not used by main.py but can be kept for other purposes
def get_car(license_plate, vehicle_detections):
    x1, y1, x2, y2, _, _ = license_plate
    for detection in vehicle_detections:
        if len(detection) >= 6:
            xcar1, ycar1, xcar2, ycar2, score, _ = detection
        else:
            continue
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return xcar1, ycar1, xcar2, y2, score
    return -1, -1, -1, -1, -1