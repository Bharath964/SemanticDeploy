import os
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from smooth_tiled_predictions import predict_img_with_smooth_windowing
from simple_multi_unet_model import jacard_coef
from skimage.measure import label as sk_label, regionprops
from patchify import patchify, unpatchify
from streamlit_drawable_canvas import st_canvas

# ------------------------
# Settings
# ------------------------
MODEL_PATH = "models/best_custom_unet_model.h5"
# MODEL_PATH = "models/InceptionResNetV2-UNet.h5"
PATCH_SIZE = 256
N_CLASSES = 6
CONVERSION_FACTOR = 0.25  # 1 pixel = 0.25 square meters

LABEL_MAPPING = {
    0: "Building",
    1: "Land",
    2: "Road",
    3: "Vegetation",
    4: "Water",
    5: "Unlabeled"
}

COLORS = {
    0: (60, 16, 152),
    1: (132, 41, 246),
    2: (110, 193, 228),
    3: (254, 221, 58),
    4: (226, 169, 41),
    5: (155, 155, 155)
}

scaler = MinMaxScaler()

def label_to_rgb(predicted_image):
    h, w = predicted_image.shape
    segmented_img = np.zeros((h, w, 3), dtype=np.uint8)
    for lbl, (r, g, b) in COLORS.items():
        segmented_img[predicted_image == lbl] = (r, g, b)
    return segmented_img

def generate_binary_mask(segmentation, target_label):
    return (segmentation == target_label).astype(np.uint8) * 255

def postprocess_mask(mask, label_id):
    if label_id == 0:  # Buildings
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    elif label_id == 4:  # Water
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

def patchwise_prediction(img, model):
    SIZE_X = (img.shape[1] // PATCH_SIZE) * PATCH_SIZE
    SIZE_Y = (img.shape[0] // PATCH_SIZE) * PATCH_SIZE
    large_img = Image.fromarray(img).crop((0, 0, SIZE_X, SIZE_Y))
    large_img = np.array(large_img)
    patches_img = patchify(large_img, (PATCH_SIZE, PATCH_SIZE, 3), step=PATCH_SIZE)
    patched_prediction = []

    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            patch = patches_img[i, j, 0, :, :, :]
            patch = scaler.fit_transform(patch.reshape(-1, 3)).reshape(patch.shape)
            patch = np.expand_dims(patch, 0)
            pred = model.predict(patch)
            pred = np.argmax(pred, axis=3)
            patched_prediction.append(pred[0])
            
    patched_prediction = np.array(patched_prediction)
    patched_prediction = np.reshape(patched_prediction,
                                    (patches_img.shape[0], patches_img.shape[1], PATCH_SIZE, PATCH_SIZE))
    return unpatchify(patched_prediction, (large_img.shape[0], large_img.shape[1]))

def smooth_prediction(img, model):
    img_scaled = scaler.fit_transform(img.reshape(-1, 3)).reshape(img.shape)
    preds = predict_img_with_smooth_windowing(
        img_scaled,
        window_size=PATCH_SIZE,
        subdivisions=2,
        nb_classes=N_CLASSES,
        pred_func=lambda batch: model.predict(batch)
    )
    return np.argmax(preds, axis=2)

def main():
    st.set_page_config(layout="wide")
    st.title("Semantic Segmentation with Dual Analysis Modes")

    @st.cache(allow_output_mutation=True)
    def load_segmentation_model():
        return load_model(MODEL_PATH, compile=False)
    model = load_segmentation_model()

    uploaded_file = st.sidebar.file_uploader("Upload Aerial Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is None:
        st.info("Please upload an image to begin.")
        return

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.subheader("Input Image")
    st.image(img_rgb, caption="Original Image", width=400)

    if st.button("Run Prediction"):
        with st.spinner("Running patch-wise prediction..."):
            patch_pred = patchwise_prediction(img_rgb, model)
            patch_rgb = label_to_rgb(patch_pred)
        with st.spinner("Running smooth tiling prediction..."):
            smooth_pred = smooth_prediction(img_rgb, model)
            smooth_rgb = label_to_rgb(smooth_pred)
        st.session_state.update({
            'patch_pred': patch_pred,
            'patch_rgb': patch_rgb,
            'smooth_pred': smooth_pred,
            'smooth_rgb': smooth_rgb,
            'prediction_done': True
        })

    if st.session_state.get('prediction_done', False):
        st.subheader("Segmentation Outputs")
        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state['patch_rgb'], caption="Patch-wise Prediction", width=300)
        with col2:
            st.image(st.session_state['smooth_rgb'], caption="Smooth Tiling Prediction", width=300)

        analysis_mode = st.radio("Select Analysis Mode:", 
                               ["Binary Mask Analysis", "Interactive Area Selection"])

        if analysis_mode == "Binary Mask Analysis":
            st.subheader("Binary Mask Analysis")
            smooth_pred = st.session_state['smooth_pred']
            unique_labels = np.unique(smooth_pred)
            label_options = [f"{i}: {LABEL_MAPPING[i]}" for i in unique_labels]
            selected_labels = st.multiselect("Select Classes for Binary Masks", options=label_options)
            selected_ids = [int(s.split(":")[0]) for s in selected_labels if s.strip()]

            if st.button("Generate Binary Masks"):
                for label_id in selected_ids:
                    mask = generate_binary_mask(smooth_pred, label_id)
                    mask = postprocess_mask(mask, label_id)
                    total_pixels = np.sum(mask == 255)
                    real_area = total_pixels * CONVERSION_FACTOR
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(mask, caption=f"{LABEL_MAPPING[label_id]} Binary Mask", use_column_width=True)
                    with col2:
                        st.markdown(f"{LABEL_MAPPING[label_id]} Total Area")
                        st.markdown(f"- Total Pixels: {total_pixels}")
                        st.markdown(f"- Real Area: {real_area:.2f} square units")

        elif analysis_mode == "Interactive Area Selection":
            st.subheader("Interactive Area Selection")
            smooth_pred = st.session_state['smooth_pred']
            smooth_rgb = st.session_state['smooth_rgb']
            original_height, original_width = smooth_rgb.shape[:2]

            # Dynamic canvas sizing with max height
            max_canvas_height = 600
            canvas_width_desired = 600
            canvas_height_desired = (original_height * canvas_width_desired) // original_width

            if canvas_height_desired > max_canvas_height:
                scale_factor = original_height / max_canvas_height
                canvas_width = int(original_width / scale_factor)
                canvas_height = max_canvas_height
            else:
                canvas_width = canvas_width_desired
                canvas_height = canvas_height_desired

            # Scale factors for mapping
            scale_factor_x = original_width / canvas_width
            scale_factor_y = original_height / canvas_height

            st.markdown("*Draw a region on the image:*")
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=2,
                stroke_color="rgb(255, 0, 0)",
                background_image=Image.fromarray(smooth_rgb),
                height=canvas_height,
                width=canvas_width,
                drawing_mode="rect",
                key="canvas",
                display_toolbar=True
            )

            st.markdown("---")  # Optional: separator line

            unique_labels = np.unique(smooth_pred)
            label_options = [f"{i}: {LABEL_MAPPING[i]}" for i in unique_labels]
            selected_class = st.selectbox("Select class for area calculation", label_options)
            label_id = int(selected_class.split(":")[0])


            if canvas_result.json_data is not None:
                    objects = canvas_result.json_data["objects"]
                    if len(objects) > 0:
                        rect = objects[-1]
                        x = rect["left"] * scale_factor_x
                        y = rect["top"] * scale_factor_y
                        w = rect["width"] * scale_factor_x
                        h = rect["height"] * scale_factor_y

                        mask = generate_binary_mask(smooth_pred, label_id)
                        mask = postprocess_mask(mask, label_id)

                        x, y, w, h = int(x), int(y), int(w), int(h)
                        cropped_mask = mask[y:y+h, x:x+w]

                        total_pixels = np.sum(cropped_mask == 255)
                        real_area = total_pixels * CONVERSION_FACTOR

                        visualization = smooth_rgb.copy()
                        cv2.rectangle(visualization, (x, y), (x + w, y + h), (255, 0, 0), 3)

                        st.image(visualization, caption="Selected Region", use_column_width=True)
                        st.markdown(f"{LABEL_MAPPING[label_id]} Area in Selection:")
                        st.markdown(f"- Total Pixels: {total_pixels}")
                        st.markdown(f"- Real Area: {real_area:.2f} square units")

if __name__ == "__main__":
    main()
