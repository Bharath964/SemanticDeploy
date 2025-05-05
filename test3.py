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
MODEL_PATH = "models/best_resnet_model (1).h5"
PATCH_SIZE = 256
N_CLASSES = 6

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
    if label_id == 4:
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

def create_mask_from_drawing(obj, scale_x, scale_y, img_shape):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    
    if obj['type'] == 'rect':
        x = int(obj['left'] * scale_x)
        y = int(obj['top'] * scale_y)
        w = int(obj['width'] * scale_x)
        h = int(obj['height'] * scale_y)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    
    elif obj['type'] == 'ellipse':
        x = int((obj['left'] + obj['width']/2) * scale_x)
        y = int((obj['top'] + obj['height']/2) * scale_y)
        rx = int(obj['width']/2 * scale_x)
        ry = int(obj['height']/2 * scale_y)
        cv2.ellipse(mask, (x, y), (rx, ry), 0, 0, 360, 255, -1)
    
    elif obj['type'] == 'path':
        points = []
        for p in obj['path']:
            if p[0] in ['M', 'L', 'C']:
                x = round(p[1] * scale_x)
                y = round(p[2] * scale_y)
                points.append((int(x), int(y)))
        
        if len(points) > 2:
            pts = np.array([points], dtype=np.int32)
            cv2.fillPoly(mask, pts, 255)
    
    return mask

def main():
    st.set_page_config(layout="wide")
    st.title("Semantic Segmentation Of Aerial Images")

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
                    total_image_pixels = mask.shape[0] * mask.shape[1]
                    percentage = (total_pixels / total_image_pixels) * 100
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(mask, caption=f"{LABEL_MAPPING[label_id]} Binary Mask", use_column_width=True)
                    with col2:
                        st.markdown(f"{LABEL_MAPPING[label_id]} Coverage")
                        st.markdown(f"- Total Pixels: {total_pixels}")
                        st.markdown(f"- Percentage of Image: {percentage:.2f}%")

        elif analysis_mode == "Interactive Area Selection":
            st.subheader("Interactive Area Selection")
            smooth_pred = st.session_state['smooth_pred']
            smooth_rgb = st.session_state['smooth_rgb']
            original_height, original_width = smooth_rgb.shape[:2]
            total_image_pixels = original_height * original_width

            drawing_mode = st.selectbox(
                "Drawing Tool:",
                ["freedraw", "rect", "ellipse"],
                index=0
            )
            
            max_canvas_height = 600
            canvas_width = 600
            canvas_height = min(int(original_height * (canvas_width / original_width)), max_canvas_height)
            scale_x = original_width / canvas_width
            scale_y = original_height / canvas_height

            st.markdown("*Draw any shape on the image:*")
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=3,
                stroke_color="rgb(255, 0, 0)",
                background_image=Image.fromarray(smooth_rgb),
                height=canvas_height,
                width=canvas_width,
                drawing_mode=drawing_mode,
                key="canvas",
                display_toolbar=True
            )

            unique_labels = np.unique(smooth_pred)
            label_options = [f"{i}: {LABEL_MAPPING[i]}" for i in unique_labels]
            selected_class = st.selectbox("Select class for area calculation", label_options)
            label_id = int(selected_class.split(":")[0])

            if canvas_result.json_data is not None:
                objects = canvas_result.json_data["objects"]
                if len(objects) > 0:
                    combined_mask = np.zeros(smooth_pred.shape[:2], dtype=np.uint8)
                    for obj in objects:
                        obj_mask = create_mask_from_drawing(
                            obj,
                            scale_x,
                            scale_y,
                            smooth_pred.shape
                        )
                        combined_mask = cv2.bitwise_or(combined_mask, obj_mask)

                    class_mask = generate_binary_mask(smooth_pred, label_id)
                    class_mask = postprocess_mask(class_mask, label_id)

                    masked_class = cv2.bitwise_and(class_mask, class_mask, mask=combined_mask)
                    total_pixels = np.sum(masked_class == 255)

                    selected_area = np.sum(combined_mask == 255)
                    percentage_in_region = (total_pixels / selected_area * 100) if selected_area > 0 else 0
                    percentage_of_image = (total_pixels / total_image_pixels) * 100

                    visualization = smooth_rgb.copy()
                    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(visualization, contours, -1, (255, 0, 0), 3)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(visualization, caption="Selected Area", use_column_width=True)
                    with col2:
                        st.markdown(f"**{LABEL_MAPPING[label_id]} Coverage Analysis**")
                        st.markdown(f"- Selected Area Pixels: {selected_area}")
                        st.markdown(f"- Class Pixels in Area: {total_pixels}")
                        st.markdown(f"- Coverage in Selected Area: {percentage_in_region:.2f}%")
                        st.markdown(f"- Percentage of Entire Image: {percentage_of_image:.2f}%")

if __name__ == "__main__":
    main()