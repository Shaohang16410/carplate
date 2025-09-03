import string
import easyocr
import cv2
import numpy as np
import streamlit as st

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
    reader = load_ocr_reader()
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

# --- NEW, ADVANCED OCR FUNCTIONS ---

def preprocess_for_ocr_advanced(image, target_height=200):
    """
    Applies a more robust series of preprocessing steps to an image to improve OCR accuracy.
    """
    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Upscale the image for better detail
    scale_factor = target_height / gray.shape[0]
    width = int(gray.shape[1] * scale_factor)
    height = int(gray.shape[0] * scale_factor)
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)

    # 3. Apply Median Blur to reduce noise while preserving edges
    blurred = cv2.medianBlur(resized, 3)

    # 4. Sharpen the image to make characters more distinct
    sharpen_kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
    sharpened = cv2.filter2D(blurred, -1, sharpen_kernel)

    # 5. Apply adaptive thresholding for binarization
    # This can be more robust than Otsu's in varying lighting
    binary_image = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # 6. Morphological operation to remove small noise
    # An 'opening' operation (erosion followed by dilation) removes small white specks
    kernel = np.ones((2, 2), np.uint8)
    opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    return opened_image


def read_license_plate_advanced(license_plate_crop, min_confidence=0.4):
    """
    Reads the license plate text using a more advanced pipeline.
    - Uses the advanced preprocessing function.
    - Provides an 'allowlist' to the OCR to constrain results.
    - Filters results based on a minimum confidence score.
    """
    reader = load_ocr_reader()

    # Apply the new advanced preprocessing
    processed_plate = preprocess_for_ocr_advanced(license_plate_crop)

    # Define the character set expected on a license plate
    allowlist = string.ascii_uppercase + string.digits

    # Perform OCR with the allowlist constraint
    detections = reader.readtext(processed_plate, allowlist=allowlist)

    if not detections:
        return None, None

    full_text = ""
    total_score = 0
    detection_count = 0

    # Filter detections by confidence and aggregate results
    for _, text, score in detections:
        if score >= min_confidence:
            full_text += text
            total_score += score
            detection_count += 1

    if detection_count == 0:
        return None, None

    avg_score = total_score / detection_count
    cleaned_text = clean_plate_text(full_text)

    # Final check on cleaned text length
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