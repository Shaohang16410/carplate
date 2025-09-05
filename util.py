import string
import easyocr
import cv2
import numpy as np
import streamlit as st

@st.cache_resource
def load_ocr_reader():
    st.info("Initializing OCR reader... This may take a moment on the first run.")
    reader = easyocr.Reader(['en'], gpu=False, recog_network='latin_g2')
    st.success("✅ OCR reader ready!")
    return reader


LICENSE_PLATE_CHARS = string.ascii_uppercase + string.digits

CHAR_CORRECTION_MAP = {
    'O': '0',
    'I': '1',
    'Z': '2',
    'S': '5',
    'B': '8',
}


def write_csv(results, output_path):
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

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    scale_factor = 2.5
    width = int(gray.shape[1] * scale_factor)
    height = int(gray.shape[0] * scale_factor)
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LANCZOS4)

    denoised = cv2.bilateralFilter(resized, 9, 75, 75)

    binary_image = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return binary_image


def clean_plate_text(text):
    base_cleaned_text = "".join(char for char in text if char in LICENSE_PLATE_CHARS).upper()

    corrected_text = ""
    for char in base_cleaned_text:
        corrected_text += CHAR_CORRECTION_MAP.get(char, char)

    return corrected_text


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.
    Applies improved preprocessing and uses an allowlist to constrain the OCR model.
    """
    reader = load_ocr_reader()
    processed_plate = preprocess_for_ocr(license_plate_crop)


    detections = reader.readtext(
        processed_plate,
        allowlist=LICENSE_PLATE_CHARS,
        paragraph=False,
        decoder='beamsearch',
        batch_size=4
    )

    if not detections:
        return None, None

    full_text = ""
    total_score = 0
    num_detections = 0
    for _, text, score in detections:
        full_text += text
        total_score += score
        num_detections += 1

    if num_detections == 0:
        return None, None

    avg_score = total_score / num_detections
    cleaned_text = clean_plate_text(full_text)

    if len(cleaned_text) >= 3:
        return cleaned_text, avg_score
    else:
        return None, None


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