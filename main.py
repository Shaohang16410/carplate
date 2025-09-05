# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
import os
import time
from ultralytics import YOLO

# Import the necessary functions from util.py
from util import get_car, read_license_plate

# --- NEW: Define default model paths ---
DEFAULT_MODEL_DIR = "models"
VEHICLE_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "vehicle-detector.pt")
LP_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "license-plate-detector.pt")

# Set page configuration
st.set_page_config(
    page_title="License Plate Recognizer",
    page_icon="🚗",
    layout="wide"
)


# --- MODEL LOADING ---
@st.cache_resource
def load_models(coco_model_file, license_plate_model_file):
    """Loads YOLO models from specified paths."""
    st.info("Loading models, please wait...")
    try:
        coco_model = YOLO(coco_model_file)
        license_plate_detector = YOLO(license_plate_model_file)
        st.success("✅ Models loaded successfully!")
        return coco_model, license_plate_detector
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()


# --- FRAME/IMAGE PROCESSING ---
def process_frame(image, coco_model, license_plate_detector):
    """
    Detects cars and license plates in a single image, performs OCR, and overlays results.
    Returns the annotated image and a list of structured detection results.
    """
    results_list = []
    annotated_image = image.copy()

    vehicle_results = coco_model(annotated_image)[0]
    vehicles = vehicle_results.boxes.data.tolist()

    lp_results = license_plate_detector(annotated_image)[0]
    license_plates = lp_results.boxes.data.tolist()

    for lp in license_plates:
        x1, y1, x2, y2, plate_score, _ = lp
        car = get_car(lp, vehicles)
        if car[0] == -1:
            continue

        xcar1, ycar1, xcar2, ycar2, car_score = car

        crop = annotated_image[int(y1):int(y2), int(x1):int(x2)]
        text, ocr_score = read_license_plate(crop)

        if text:
            cv2.rectangle(annotated_image, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0), 2)
            cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(annotated_image, text, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            results_list.append({
                "text": text,
                "car_score": car_score,
                "plate_bbox_score": plate_score,
                "ocr_score": ocr_score
            })
    return annotated_image, results_list


# --- MAIN APPLICATION ---
def main():
    st.title("🚗 License Plate Recognition App")
    st.markdown("This app detects and recognizes license plates from an uploaded image.")

    # Initialize models as None
    coco_model = None
    license_plate_detector = None

    # --- UPDATED LOGIC: CHECK FOR DEFAULT MODELS OR USE UPLOADERS ---
    st.sidebar.header("Model Configuration")

    # Check if default models exist in the 'models' directory
    if os.path.exists(VEHICLE_MODEL_PATH) and os.path.exists(LP_MODEL_PATH):
        st.sidebar.success(f"✅ Default models found in `/{DEFAULT_MODEL_DIR}` folder.")
        coco_model, license_plate_detector = load_models(VEHICLE_MODEL_PATH, LP_MODEL_PATH)
    else:
        st.sidebar.warning(f"Default models not found in `/{DEFAULT_MODEL_DIR}`.")
        st.sidebar.info("Please upload model files to proceed.")

        uploaded_coco_model = st.sidebar.file_uploader("Upload Vehicle Detection Model (.pt)", type=['pt'])
        uploaded_lp_model = st.sidebar.file_uploader("Upload License Plate Model (.pt)", type=['pt'])

        if uploaded_coco_model is not None and uploaded_lp_model is not None:
            # Save uploaded models to temporary files
            with open("temp_coco.pt", "wb") as f:
                f.write(uploaded_coco_model.getbuffer())
            with open("temp_lp.pt", "wb") as f:
                f.write(uploaded_lp_model.getbuffer())

            coco_model, license_plate_detector = load_models("temp_coco.pt", "temp_lp.pt")

            # Clean up temp files after loading
            os.remove("temp_coco.pt")
            os.remove("temp_lp.pt")

    # --- Run main app only if models are loaded ---
    if coco_model and license_plate_detector:
        st.header("Upload an Image for Processing")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

            if st.button("Process Image"):
                start_time = time.time()
                with st.spinner("Processing image..."):
                    processed_image, results = process_frame(image, coco_model, license_plate_detector)
                end_time = time.time()
                processing_time = end_time - start_time

                with col2:
                    st.subheader("Processed Image")
                    st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), use_column_width=True)
                    st.success(f"**Processing Time:** {processing_time:.2f} seconds")

                st.subheader("Detection Results")
                if results:
                    for i, res in enumerate(results):
                        st.success(f"**License Plate {i + 1}:** {res['text']}")
                        st.write(f"  - Car Confidence: {res['car_score'] * 100:.2f}%")
                        st.write(f"  - Plate Confidence: {res['plate_bbox_score'] * 100:.2f}%")
                        st.write(f"  - OCR Confidence: {res['ocr_score'] * 100:.2f}%")
                else:
                    st.warning("No license plates detected in the image.")
    else:
        st.warning(
            "Models not loaded. Please place default models in the 'models' folder or upload them via the sidebar.")


if __name__ == "__main__":
    main()