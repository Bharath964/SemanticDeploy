"""
Prediction using smooth tiling with binary mask selection and per-object area calculation

This script loads a trained segmentation model and performs prediction on an input aerial 
image using both patch-wise (non-overlapping) prediction and smooth blending. It then displays 
the unique labels present in the segmentation and repeatedly asks the user to select one or more 
classes for which binary masks will be generated.

For each selected label, the script generates a binary mask, finds each connected component 
(object) in that mask, and then sums the pixel areas of all such objects. The pixel area is then 
multiplied by a conversion factor to yield the real‑world area, which is printed to the console.

Color mapping:
    0: Building
    1: Land
    2: Road
    3: Vegetation
    4: Water
    5: Unlabeled

Original code from:
https://github.com/Vooban/Smoothly-Blend-Image-Patches (MIT License)
Modified for binary mask functionality, interactive selection, and per-object area calculation.
"""

import os  # for file path handling
import cv2
import numpy as np
import gc
import matplotlib.pyplot as plt
from patchify import patchify, unpatchify
from PIL import Image
import segmentation_models as sm
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from smooth_tiled_predictions import predict_img_with_smooth_windowing
from simple_multi_unet_model import jacard_coef
from tqdm import tqdm
from skimage.measure import label as sk_label, regionprops

# --------------------------
# SETTINGS & PATHS
# --------------------------
# Adjust these paths as needed.
img_path = r""
mask_path = r"Semantic segmentation dataset\Tile 8\masks\image_part_001.png"  # Ground-truth mask (if available)
output_dir = os.path.expanduser(r"~/Semantic/models")

# Conversion factor (e.g., 1 pixel corresponds to 0.25 square meters – adjust accordingly)
conversion_factor = 0.25

# Load the test image and (optionally) the ground-truth mask.
img = cv2.imread(img_path, 1)
if img is None:
    raise ValueError(f"Could not load the image at {img_path}. Check the file path and file integrity.")
original_mask = cv2.imread(mask_path, 1)
if original_mask is None:
    print(f"Warning: Could not load the mask at {mask_path}. Continuing without ground-truth mask.")
    original_mask = np.zeros_like(img)
else:
    original_mask = cv2.cvtColor(original_mask, cv2.COLOR_BGR2RGB)

# Load the trained segmentation model.
from keras.models import load_model
model = load_model(r"models\best_custom_unet_model.h5", compile=False)

# Prediction settings
patch_size = 256
n_classes = 6

# --------------------------
# 1. Prediction: Patch-wise (Non-Smooth)
# --------------------------
SIZE_X = (img.shape[1] // patch_size) * patch_size
SIZE_Y = (img.shape[0] // patch_size) * patch_size
large_img = Image.fromarray(img)
large_img = large_img.crop((0, 0, SIZE_X, SIZE_Y))
large_img = np.array(large_img)

patches_img = patchify(large_img, (patch_size, patch_size, 3), step=patch_size)
patched_prediction = []
for i in range(patches_img.shape[0]):
    for j in range(patches_img.shape[1]):
        single_patch_img = patches_img[i, j, 0, :, :, :]
        single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
        single_patch_img = np.expand_dims(single_patch_img, axis=0)  # add batch dimension
        pred = model.predict(single_patch_img)
        pred = np.argmax(pred, axis=3)
        patched_prediction.append(pred[0, :, :])
patched_prediction = np.array(patched_prediction)
patched_prediction = np.reshape(patched_prediction, (patches_img.shape[0], patches_img.shape[1], patch_size, patch_size))
unpatched_prediction = unpatchify(patched_prediction, (large_img.shape[0], large_img.shape[1]))

plt.figure(figsize=(6, 6))
plt.imshow(unpatched_prediction, cmap="jet")
plt.title("Non-Smooth Prediction (Patch-wise)")
plt.axis('off')
plt.show()

# --------------------------
# 2. Prediction: Smooth Blending
# --------------------------
input_img = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape)
predictions_smooth = predict_img_with_smooth_windowing(
    input_img,
    window_size=patch_size,
    subdivisions=2,   # Controls overlap; must be even.
    nb_classes=n_classes,
    pred_func=lambda batch: model.predict(batch)
)
final_prediction = np.argmax(predictions_smooth, axis=2)

# --------------------------
# 3. Convert Label Predictions to RGB for Visualization
# --------------------------
def label_to_rgb(predicted_image):
    # Define color mapping.
    Building = np.array([60, 16, 152])
    Land = np.array([132, 41, 246])
    Road = np.array([110, 193, 228])
    Vegetation = np.array([254, 221, 58])
    Water = np.array([226, 169, 41])
    Unlabeled = np.array([155, 155, 155])
    
    segmented_img = np.empty((predicted_image.shape[0], predicted_image.shape[1], 3), dtype=np.uint8)
    segmented_img[predicted_image == 0] = Building
    segmented_img[predicted_image == 1] = Land
    segmented_img[predicted_image == 2] = Road
    segmented_img[predicted_image == 3] = Vegetation
    segmented_img[predicted_image == 4] = Water
    segmented_img[predicted_image == 5] = Unlabeled
    return segmented_img

prediction_with_smooth_blending = label_to_rgb(final_prediction)
prediction_without_smooth_blending = label_to_rgb(unpatched_prediction)

plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.title('Testing Image')
plt.imshow(img)
plt.subplot(222)
plt.title('Testing Label')
plt.imshow(original_mask)
plt.subplot(223)
plt.title('Patch-wise Prediction')
plt.imshow(prediction_without_smooth_blending)
plt.subplot(224)
plt.title('Smooth Blending Prediction')
plt.imshow(prediction_with_smooth_blending)
plt.show()

# --------------------------
# 4. Interactive Binary Mask Generation & Total Area Calculation
# --------------------------
# Mapping for human-readable class names.
label_mapping = {
    0: "Building",
    1: "Land",
    2: "Road",
    3: "Vegetation",
    4: "Water",
    5: "Unlabeled"
}

def generate_binary_mask(segmentation, target_label):
    """Generate a binary mask (values 0 and 255) for the given target label."""
    return (segmentation == target_label).astype(np.uint8) * 255

# Display available labels.
unique_labels = np.unique(final_prediction)
print("\nUnique labels in predicted image:", unique_labels)
print("Available classes:")
for lbl in unique_labels:
    print(f"  {lbl}: {label_mapping.get(lbl, 'Unknown')}")

# Interactive loop to allow repeated selections.
while True:
    user_input = input("\nEnter label numbers (comma separated) to generate masks & calculate total area (or 'q' to quit): ")
    if user_input.lower() == "q":
        print("Exiting the area calculation loop.")
        break
    
    try:
        selected_labels = [int(x.strip()) for x in user_input.split(",")]
    except ValueError:
        print("Error parsing input. Please enter label indices like 0,3,4 or type 'q' to quit.")
        continue
    
    # Generate a combined binary mask for visualization.
    combined_binary_mask = np.zeros(final_prediction.shape, dtype=np.uint8)
    for lbl in selected_labels:
        if lbl in unique_labels:
            combined_binary_mask |= (final_prediction == lbl).astype(np.uint8)
        else:
            print(f"Label {lbl} not present in prediction. Skipping.")
    combined_binary_mask *= 255
    combined_mask_save_path = os.path.join(output_dir, "combined_binary_mask.png")
    cv2.imwrite(combined_mask_save_path, combined_binary_mask)
    print(f"Saved combined binary mask at: {combined_mask_save_path}")
    
    plt.figure()
    plt.imshow(combined_binary_mask, cmap="gray")
    plt.title("Combined Binary Mask (Selected Classes)")
    plt.axis('off')
    plt.show(block=False)
    
    # For each selected label, generate its binary mask, compute per-object areas, and print the total area.
    for lbl in selected_labels:
        if lbl not in unique_labels:
            continue
        class_name = label_mapping.get(lbl, f"Label {lbl}")
        binary_mask = generate_binary_mask(final_prediction, lbl)
        # Label connected components.
        labeled_mask = sk_label(binary_mask)
        regions = regionprops(labeled_mask)
        print(f"\nAreas for objects in '{class_name}' (Label {lbl}):")
        total_pixels = 0
        if not regions:
            print("  No objects found.")
        else:
            for region in regions:
                area_pixels = region.area
                total_pixels += area_pixels
                real_area = area_pixels * conversion_factor
                print(f"  - Object #{region.label}: {area_pixels} pixels, {real_area:.2f} square units")
            total_real_area = total_pixels * conversion_factor
            print(f"Total area for {class_name}: {total_pixels} pixels, {total_real_area:.2f} square units")
    
    cont = input("\nPress Enter to select more labels (or type 'q' to quit): ")
    if cont.lower() == "q":
        break
    plt.close('all')

print("Interactive binary mask generation and area calculation finished.")
