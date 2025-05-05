import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
from PIL import Image
import segmentation_models as sm
from tensorflow.keras.metrics import MeanIoU
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, 
                            f1_score, accuracy_score, average_precision_score)
from tensorflow.keras.utils import to_categorical
import random

#======================= CONFIGURATION =======================#
root_directory = 'Semantic segmentation dataset/'
patch_size = 256
output_dir = r'C:\Users\Bharath\Desktop\Major\Code\try5WebApp\models'

# C:\Users\Bharath\Desktop\Major\Code\try5WebApp\models\best_custom_unet_model.h5
# C:\Users\Bharath\Desktop\Major\Code\try5WebApp\models\best_resnet_model (1).h5
LABEL_MAPPING = {
    0: "Building", 1: "Land", 2: "Road",
    3: "Vegetation", 4: "Water", 5: "Unlabeled"
}
#=============================================================#


# Define RGB values first!
Building = np.array([60, 16, 152])    # #3C1098
Land = np.array([132, 41, 246])       # #8429F6
Road = np.array([110, 193, 228])      # #6EC1E4
Vegetation = np.array([254, 221, 58]) # #FEDD3A
Water = np.array([226, 169, 41])      # #E2A929
Unlabeled = np.array([155, 155, 155]) # #9B9B9B

#====================== DATA PROCESSING ======================#
# Initialize scaler
scaler = MinMaxScaler()

# Image processing
image_dataset = []  
for path, subdirs, files in os.walk(root_directory):
    dirname = path.split(os.path.sep)[-1]
    if dirname == 'images':
        for image_name in [f for f in files if f.endswith(".jpg")]:
            image = cv2.imread(os.path.join(path, image_name), 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            SIZE_X = (image.shape[1]//patch_size)*patch_size
            SIZE_Y = (image.shape[0]//patch_size)*patch_size
            image = Image.fromarray(image).crop((0, 0, SIZE_X, SIZE_Y))
            image = np.array(image)
            
            # Extract and scale patches
            patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)
            for i in range(patches_img.shape[0]):
                for j in range(patches_img.shape[1]):
                    patch = patches_img[i,j,0]
                    patch = scaler.fit_transform(patch.reshape(-1, 3)).reshape(patch.shape)
                    image_dataset.append(patch)  # Removed [0]

# Mask processing
mask_dataset = []  
for path, subdirs, files in os.walk(root_directory):
    dirname = path.split(os.path.sep)[-1]
    if dirname == 'masks':
        for mask_name in [f for f in files if f.endswith(".png")]:
            mask = cv2.imread(os.path.join(path, mask_name), 1)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            SIZE_X = (mask.shape[1]//patch_size)*patch_size
            SIZE_Y = (mask.shape[0]//patch_size)*patch_size
            mask = Image.fromarray(mask).crop((0, 0, SIZE_X, SIZE_Y))
            mask = np.array(mask)
            
            # Extract mask patches
            patches_mask = patchify(mask, (patch_size, patch_size, 3), step=patch_size)
            for i in range(patches_mask.shape[0]):
                for j in range(patches_mask.shape[1]):
                   mask_dataset.append(patches_mask[i,j,0,:,:,:])  # Correct full patch indexing
  # Removed [0]

# Convert to arrays
image_dataset = np.array(image_dataset)
mask_dataset = np.array(mask_dataset)

# Building = np.array(tuple(int('#3C1098'.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)))
# Land = np.array(tuple(int('#8429F6'.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)))
# Road = np.array(tuple(int('#6EC1E4'.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)))
# Vegetation = np.array(tuple(int('FEDD3A'.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)))
# Water = np.array(tuple(int('E2A929'.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)))
# Unlabeled = np.array(tuple(int('#9B9B9B'.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)))

# Convert RGB masks to class labels
def rgb_to_2D_label(label):
    """
    Suply our labale masks as input in RGB format. 
    Replace pixels with specific RGB values ...
    """
    label_seg = np.zeros(label.shape[:2], dtype=np.uint8)
    label_seg [np.all(label == Building,axis=-1)] = 0
    label_seg [np.all(label==Land,axis=-1)] = 1
    label_seg [np.all(label==Road,axis=-1)] = 2
    label_seg [np.all(label==Vegetation,axis=-1)] = 3
    label_seg [np.all(label==Water,axis=-1)] = 4
    label_seg [np.all(label==Unlabeled,axis=-1)] = 5
    
    # label_seg = label_seg[:,:,0]  #Just take the first channel, no need for all 3 channels
    return label_seg

labels = []
for i in range(mask_dataset.shape[0]):
    label = rgb_to_2D_label(mask_dataset[i])
    labels.append(label)    


labels = np.array([rgb_to_2D_label(m) for m in mask_dataset])
labels = np.expand_dims(labels, axis=3)
labels_cat = to_categorical(labels, num_classes=len(LABEL_MAPPING))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    image_dataset, labels_cat, test_size=0.20, random_state=42
)
#=============================================================#

#===================== EVALUATION SECTION ====================#
def evaluate_model(model_path):
    # Load model
    model = load_model(model_path, compile=False)

    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_argmax = np.argmax(y_pred, axis=3)
    y_test_argmax = np.argmax(y_test, axis=3)
    
    # Calculate metrics
    def calculate_metrics(y_true, y_pred, n_classes):
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        mean_iou_metric = MeanIoU(num_classes=n_classes)
        mean_iou_metric.update_state(y_true_flat, y_pred_flat)
        mean_iou = mean_iou_metric.result().numpy()

        return {
            'Accuracy': accuracy_score(y_true_flat, y_pred_flat),
            'Precision': precision_score(y_true_flat, y_pred_flat, average='weighted'),
            'Recall': recall_score(y_true_flat, y_pred_flat, average='weighted'),
            'F1': f1_score(y_true_flat, y_pred_flat, average='weighted'),
            'mAP@0.5': np.mean([average_precision_score((y_true_flat == i), (y_pred_flat == i)) 
                          for i in range(n_classes)]),
            'Mean IoU': mean_iou
        }
    
    metrics = calculate_metrics(y_test_argmax, y_pred_argmax, len(LABEL_MAPPING))
    
    print(f"\n=== Evaluation Metrics for {os.path.basename(model_path)} ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    # Visualization
    test_img_number = random.randint(0, len(X_test))
    test_img = X_test[test_img_number]
    ground_truth = y_test_argmax[test_img_number]
    prediction = np.argmax(model.predict(np.expand_dims(test_img, 0)), axis=3)[0]
    
    display_img = (test_img * 255).astype(np.uint8)

    plt.figure(figsize=(12, 8))
    plt.subplot(131)
    plt.title('Input Image')
    plt.imshow(display_img)
    # plt.imshow(test_img)
    plt.subplot(132)
    plt.title('Ground Truth')
    plt.imshow(ground_truth)
    plt.subplot(133)
    plt.title('Prediction')
    plt.imshow(prediction)
    plt.show()
    
    return metrics

# List of models to evaluate
model_paths = [
    os.path.join(output_dir, 'best_custom_unet_model.h5'),
    os.path.join(output_dir, 'best_resnet_model (1).h5')
]

# Run evaluation for all models
for path in model_paths:
    if os.path.exists(path):
        evaluate_model(path)
    else:
        print(f"Model not found: {path}")
#=============================================================#