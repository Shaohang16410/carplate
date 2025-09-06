# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
import os
import time
from ultralytics import YOLO

# Import the necessary functions from util.py
from util import get_car, read_license_plate

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
        # --- END OF FIX ---

        crop = annotated_image[int(y1):int(y2), int(x1):int(x2)]
        text, ocr_score = read_license_plate(crop)

        if text:
            # Now xcar2 and ycar2 are defined and can be used here
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
    st.markdown("Upload your models and an image to detect and recognize license plates.")

    # --- SIDEBAR FOR MODEL UPLOAD ---
    st.sidebar.header("Model Configuration")
    st.sidebar.info("You must upload both models to proceed.")

    uploaded_coco_model = st.sidebar.file_uploader("Upload Vehicle Detection Model (.pt)", type=['pt'])
    uploaded_lp_model = st.sidebar.file_uploader("Upload License Plate Model (.pt)", type=['pt'])

    # --- START: New Section for Model Performance Uploads ---
    st.sidebar.markdown("---")
    with st.sidebar.expander("Upload Model Performance Images (Optional)"):
        uploaded_results_img = st.file_uploader("Upload Training Results Chart (e.g., results.png)", type=["png", "jpg", "jpeg"])
        uploaded_cm_img = st.file_uploader("Upload Confusion Matrix (e.g., confusion_matrix.png)", type=["png", "jpg", "jpeg"])
        uploaded_labels_img = st.file_uploader("Upload Labels Distribution (e.g., labels.jpg)", type=["png", "jpg", "jpeg"])
    # --- END: New Section for Model Performance Uploads ---


    if uploaded_coco_model is not None and uploaded_lp_model is not None:
        coco_model_path = "temp_coco.pt"
        license_plate_model_path = "temp_lp.pt"
        with open(coco_model_path, "wb") as f:
            f.write(uploaded_coco_model.getbuffer())
        with open(license_plate_model_path, "wb") as f:
            f.write(uploaded_lp_model.getbuffer())

        coco_model, license_plate_detector = load_models(coco_model_path, license_plate_model_path)

        st.header("Upload an Image for Detection")
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

        # --- START: New Section for Displaying Model Performance ---
        st.markdown("---")
        st.header("License Plate Model Performance Analysis")

        if uploaded_results_img or uploaded_cm_img or uploaded_labels_img:
            st.info("Displaying uploaded model performance charts.")

            if uploaded_results_img:
                st.subheader("Training & Validation Curves")
                st.image(uploaded_results_img, caption="Loss and metrics curves over training epochs.", use_column_width=True)

            col1, col2 = st.columns(2)
            with col1:
                if uploaded_cm_img:
                    st.subheader("Confusion Matrix")
                    st.image(uploaded_cm_img, caption="Model's confusion matrix on the validation set.", use_column_width=True)

            with col2:
                if uploaded_labels_img:
                    st.subheader("Training Data Labels")
                    st.image(uploaded_labels_img, caption="Distribution of labels in the training dataset.", use_column_width=True)
        else:
            st.warning("No model performance images were uploaded. To see them, upload the relevant files in the sidebar.")
        # --- END: New Section for Displaying Model Performance ---


    else:
        st.warning("Please upload both model files using the sidebar to continue.")


if __name__ == "__main__":
    main()