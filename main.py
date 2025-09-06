# main.py
import streamlit as st
import cv2
import numpy as np
import os
import time
from ultralytics import YOLO

# Import the necessary functions from util.py
from util import read_license_plate  # get_car is no longer needed

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
        vehicle_detector = YOLO(coco_model_file)
        license_plate_detector = YOLO(license_plate_model_file)
        st.success("✅ Models loaded successfully!")
        return vehicle_detector, license_plate_detector
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        st.stop()


# --- FRAME/IMAGE PROCESSING (Updated Logic) ---
def process_frame(image, vehicle_detector, license_plate_detector):
    """
    Processes a single image frame to detect vehicles, then license plates within them.
    This function now follows the two-stage pipeline from the diagram.
    """
    results_list = []
    annotated_image = image.copy()

    # 1. Detect Vehicles
    vehicle_results = vehicle_detector(image)[0]
    vehicles = vehicle_results.boxes.data.tolist()

    # COCO class IDs for vehicles: 2=car, 3=motorcycle, 5=bus, 7=truck
    vehicle_class_ids = [2, 3, 5, 7]

    for vehicle in vehicles:
        xcar1, ycar1, xcar2, ycar2, car_score, class_id = vehicle

        # Filter for vehicle classes
        if int(class_id) not in vehicle_class_ids:
            continue

        # 2. Crop Each Vehicle
        car_crop = image[int(ycar1):int(ycar2), int(xcar1):int(xcar2)]
        if car_crop.size == 0:
            continue

        # 3. Detect License Plates within the vehicle crop
        lp_results = license_plate_detector(car_crop)[0]
        license_plates = lp_results.boxes.data.tolist()

        for lp in license_plates:
            # Coordinates are relative to the car_crop
            x1_rel, y1_rel, x2_rel, y2_rel, plate_score, _ = lp

            # 4. Crop the license plate
            lp_crop = car_crop[int(y1_rel):int(y2_rel), int(x1_rel):int(x2_rel)]
            if lp_crop.size == 0:
                continue

            # 5. Perform OCR
            text, ocr_score = read_license_plate(lp_crop)

            if text:
                # Convert relative license plate coordinates to absolute image coordinates
                x1_abs = int(xcar1 + x1_rel)
                y1_abs = int(ycar1 + y1_rel)
                x2_abs = int(xcar1 + x2_rel)
                y2_abs = int(ycar1 + y2_rel)

                # Draw bounding boxes on the annotated image
                cv2.rectangle(annotated_image, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0), 2)
                cv2.rectangle(annotated_image, (x1_abs, y1_abs), (x2_abs, y2_abs), (0, 0, 255), 2)
                cv2.putText(annotated_image, text, (x1_abs, y1_abs - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                results_list.append({
                    "text": text,
                    "car_score": car_score,
                    "plate_bbox_score": plate_score,
                    "ocr_score": ocr_score
                })
                # Break after finding the first plate in a car for simplicity
                # Remove this break if a car can have multiple plates you want to detect
                break

    return annotated_image, results_list


# --- MAIN APPLICATION ---
def main():
    st.title("🚗 License Plate Recognition App")
    st.markdown("Upload your models and an image to detect and recognize license plates.")

    # --- SIDEBAR FOR MODEL UPLOAD ---
    st.sidebar.header("Model Configuration")
    st.sidebar.info("You must upload both models to proceed.")

    uploaded_coco_model = st.sidebar.file_uploader("Upload Vehicle Detection Model (.pt)", type=['pt'])
    uploaded_lp_model = st.sidebar.file_uploader("Upload License Plate Model (.pt)", type=['pt'])

    if uploaded_coco_model is not None and uploaded_lp_model is not None:
        coco_model_path = "temp_coco.pt"
        license_plate_model_path = "temp_lp.pt"
        with open(coco_model_path, "wb") as f:
            f.write(uploaded_coco_model.getbuffer())
        with open(license_plate_model_path, "wb") as f:
            f.write(uploaded_lp_model.getbuffer())

        vehicle_detector, license_plate_detector = load_models(coco_model_path, license_plate_model_path)

        st.header("Upload an Image")
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
                    processed_image, results = process_frame(image, vehicle_detector, license_plate_detector)
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
        st.warning("Please upload both model files using the sidebar to continue.")


if __name__ == "__main__":
    main()