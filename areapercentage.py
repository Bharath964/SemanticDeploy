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
# MODEL_PATH = "models/best_custom_unet_model.h5"
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

# COLORS = {
#     0: (169, 169, 169),  # Buildings in Dark Grey
#     1: (139, 69, 19),    # Land in Brown
#     2: (135, 206, 235),  # Roads in Sky Blue
#     3: (34, 139, 34),    # Vegetation in Green
#     4: (70, 130, 180),   # Water in Water Blue
#     5: (255, 255, 255)   # Unlabeled in White
# }

# An image has pixel values like this:
# Red, green, blue values are usually in the range 0 to 255
# But many machine learning models (especially neural networks ) work better when input values are between 0 and 1.
# So, we scale each pixel using this formula:
# scaled_value = (value - min) / (max - min)
# | Without Scaling   | With Scaling               |
# | ----------------- | -------------------------- |
# | Pixel value = 200 | Pixel value becomes ≈ 0.78 |
# | Pixel value = 50  | Pixel value becomes ≈ 0.20 |
# | Pixel value = 255 | Pixel value becomes 1.0    |

scaler = MinMaxScaler()


# This function converts a segmentation output (which is just numbers for each class) into a colorful image so humans can understand it.
# What does predicted_image look like?
# Example:
# predicted_image = 
# [[0, 0, 1],
#  [2, 2, 3],
#  [4, 5, 5]]
# Each number here represents a class label:
# 0 = Building
# 1 = Land
# 2 = Road
# ...etc.
# But we can’t see this properly as an image — it's just numbers. So It replaces each number with a specific RGB color:
def label_to_rgb(predicted_image):
    h, w = predicted_image.shape
    segmented_img = np.zeros((h, w, 3), dtype=np.uint8)
    for lbl, (r, g, b) in COLORS.items():
        segmented_img[predicted_image == lbl] = (r, g, b)
    return segmented_img

def generate_binary_mask(segmentation, target_label):
    return (segmentation == target_label).astype(np.uint8) * 255

# def generate_binary_mask(segmentation, target_label):
#     h, w = segmentation.shape
#     colored_mask = np.zeros((h, w, 3), dtype=np.uint8)  # All black initially
#     color = COLORS[target_label]  # Get the RGB color for the label
#     colored_mask[segmentation == target_label] = color  # Apply only to selected class
#     return colored_mask


# Goal of postprocess_mask:
# This function cleans up the mask that you get after segmentation. The raw mask might look broken,
#  noisy, or have tiny holes or dots — and we want to fix that.

# You have an area (say, buildings) that should be solid — but it's broken into pieces or has gaps or tiny spots around it.
# This function says:
# “If it's a building, let me fill those small holes and connect nearby building pieces.”
# “If it's water, let me remove tiny specks that shouldn’t be there.”

def postprocess_mask(mask, label_id):
    # if label_id == 0:  # Buildings
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    if label_id == 4:
    # elif label_id == 4:  # Water
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

# def postprocess_mask(mask, label_id):
#     if label_id == 0:  # Buildings - fill gaps, connect nearby pixels
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

#     elif label_id == 1:  # Land - minimal changes, maybe remove small holes
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

#     elif label_id == 2:  # Road - clean narrow areas
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

#     elif label_id == 3:  # Vegetation - remove small noise
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

#     elif label_id == 4:  # Water - remove speckles
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

#     elif label_id == 5:  # Unlabeled - clear scattered pixels
#         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

#     return mask




# This function patchwise_prediction:
# Divides a big image into small patches of size 256×256.
# Normalizes each patch using MinMaxScaler.
# Predicts the segmentation result for each patch using the model.
# Stitches the predictions of all patches back into one full-size output image.

def patchwise_prediction(img, model):
    # This crops the input image size to the nearest multiple of 256 (PATCH_SIZE).
    # Why? Because the model works with fixed-size patches.
    SIZE_X = (img.shape[1] // PATCH_SIZE) * PATCH_SIZE
    SIZE_Y = (img.shape[0] // PATCH_SIZE) * PATCH_SIZE

    # Converts the image into a PIL(pillow) Image just to crop it.
    # Pillow has a simple .crop() function that makes it easy to crop an image to the required size.
    # OpenCV could also do it, but Pillow is cleaner for such basic image manipulations.
    # Then back to NumPy array for further processing.
    large_img = Image.fromarray(img).crop((0, 0, SIZE_X, SIZE_Y))
    large_img = np.array(large_img)

    # This splits the image into non-overlapping patches (256×256 RGB).
    patches_img = patchify(large_img, (PATCH_SIZE, PATCH_SIZE, 3), step=PATCH_SIZE)

    patched_prediction = []
    # Loops over each patch.
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            patch = patches_img[i, j, 0, :, :, :]

            # Normalizes pixel values in the patch to range [0, 1] using MinMaxScaler.
            patch = scaler.fit_transform(patch.reshape(-1, 3)).reshape(patch.shape)

            # Adds a batch dimension → (1, 256, 256, 3).
            # Feeds to the model.
            # argmax gives the predicted label (0–5) for each pixel.
            patch = np.expand_dims(patch, 0)
            pred = model.predict(patch)
            pred = np.argmax(pred, axis=3)
            patched_prediction.append(pred[0])
    
    # Reshapes the list of predicted patches into a grid again.
    patched_prediction = np.array(patched_prediction)
    patched_prediction = np.reshape(patched_prediction,
                                    (patches_img.shape[0], patches_img.shape[1], PATCH_SIZE, PATCH_SIZE))
    
    # Stitches back all patches into one large image with per-pixel labels.
    return unpatchify(patched_prediction, (large_img.shape[0], large_img.shape[1]))

# def smooth_prediction(img, model):
#     img_scaled = scaler.fit_transform(img.reshape(-1, 3)).reshape(img.shape)
#     preds = predict_img_with_smooth_windowing(
#         img_scaled,
#         window_size=PATCH_SIZE,
#         subdivisions=2,
#         nb_classes=N_CLASSES,
#         pred_func=lambda batch: model.predict(batch)
#     )
#     return np.argmax(preds, axis=2)

def smooth_prediction(img, model):
    # Crop image to compatible dimensions
    SIZE_X = (img.shape[1] // PATCH_SIZE) * PATCH_SIZE
    SIZE_Y = (img.shape[0] // PATCH_SIZE) * PATCH_SIZE
    img_cropped = Image.fromarray(img).crop((0, 0, SIZE_X, SIZE_Y))
    img_cropped = np.array(img_cropped)
    
    # Scale image
    img_scaled = scaler.fit_transform(img_cropped.reshape(-1, 3)).reshape(img_cropped.shape)
    
    # Get initial prediction using smooth tiling
    initial_preds = predict_img_with_smooth_windowing(
        img_scaled,
        window_size=PATCH_SIZE,
        subdivisions=2,
        nb_classes=N_CLASSES,
        pred_func=lambda batch: model.predict(batch)
    )  # FIXED: Added closing parenthesis here
    initial_pred = np.argmax(initial_preds, axis=2)
    road_mask = (initial_pred == 2)

    # Get refined predictions with different parameters
    refined_preds = predict_img_with_smooth_windowing(
        img_scaled,
        window_size=PATCH_SIZE,
        subdivisions=4,  # Increased overlap for non-road features
        nb_classes=N_CLASSES,
        pred_func=lambda batch: model.predict(batch)
    )
    refined_labels = np.argmax(refined_preds, axis=2)

    # Combine results: keep original roads, use refined predictions for others
    final_pred = np.where(road_mask, initial_pred, refined_labels)
    
    return final_pred

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
            #  gets the predicted labels for the image (in the form of an array of label IDs).
            smooth_pred = st.session_state['smooth_pred']
            # finds all unique labels in the prediction (e.g., each label represents a different class).
            unique_labels = np.unique(smooth_pred)
            label_options = [f"{i}: {LABEL_MAPPING[i]}" for i in unique_labels]
            selected_labels = st.multiselect("Select Classes for Binary Masks", options=label_options)
            selected_ids = [int(s.split(":")[0]) for s in selected_labels if s.strip()]

            if st.button("Generate Binary Masks"):
                for label_id in selected_ids:
                    mask = generate_binary_mask(smooth_pred, label_id)
                    mask = postprocess_mask(mask, label_id)
                    total_pixels = np.sum(mask == 255)

                    # mask.shape[0] * mask.shape[1] gives the total number of pixels in the image (height * width of the mask).
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
            # In Python, especially with NumPy arrays (which is how images are usually stored), the shape property
            #  returns the dimensions of the array:
            original_height, original_width = smooth_rgb.shape[:2]
            total_image_pixels = original_height * original_width  # Added total image pixels

            # Dynamic canvas sizing
            max_canvas_height = 600
            canvas_width = 600
            canvas_height = min(int(original_height * (canvas_width / original_width)), max_canvas_height)

            # Scale factors for coordinate mapping
            scale_x = original_width / canvas_width
            scale_y = original_height / canvas_height

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

            unique_labels = np.unique(smooth_pred)
            label_options = [f"{i}: {LABEL_MAPPING[i]}" for i in unique_labels]
            selected_class = st.selectbox("Select class for area calculation", label_options)
            label_id = int(selected_class.split(":")[0])

            if canvas_result.json_data is not None:
                objects = canvas_result.json_data["objects"]
                if len(objects) > 0:
                    rect = objects[-1]
                    x = int(rect["left"] * scale_x)
                    y = int(rect["top"] * scale_y)
                    w = int(rect["width"] * scale_x)
                    h = int(rect["height"] * scale_y)

                    mask = generate_binary_mask(smooth_pred, label_id)
                    mask = postprocess_mask(mask, label_id)

                    cropped_mask = mask[y:y+h, x:x+w]
                    
                    # Calculate both percentages
                    total_pixels = np.sum(cropped_mask == 255)
                    region_area = w * h
                    percentage_in_region = (total_pixels / region_area) * 100 if region_area > 0 else 0.0
                    percentage_of_image = (total_pixels / total_image_pixels) * 100  # New calculation

                    visualization = smooth_rgb.copy()
                    cv2.rectangle(visualization, (x, y), (x + w, y + h), (255, 0, 0), 3)

                    st.image(visualization, caption="Selected Region", use_column_width=True)
                    st.markdown(f"{LABEL_MAPPING[label_id]} Coverage:")
                    st.markdown(f"- Pixels in Region: {total_pixels}")
                    # st.markdown(f"- Percentage **within Region**: {percentage_in_region:.2f}%")
                    st.markdown(f"- Percentage **of Entire Image**: {percentage_of_image:.2f}%")  # New line

if __name__ == "__main__":
    main()