import os
import cv2
import numpy as np

# Setup model save directory
output_dir = os.path.expanduser('~/Semantic/models')
os.makedirs(output_dir, exist_ok=True)
print(f"Model save directory: {output_dir}")


from matplotlib import pyplot as plt
from patchify import patchify #divide into smaller patches
from PIL import Image #crop func
import segmentation_models as sm
from tensorflow.keras.metrics import MeanIoU
from keras.callbacks import ModelCheckpoint
from keras.models import save_model


from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()

root_directory = 'Semantic segmentation dataset/'

patch_size = 256

#Read images from repsective 'images' subdirectory
#As all images are of ddifferent size we have 2 options, either resize or crop
#But, some images are too large and some small. Resizing will change the size of real objects.
#Therefore, we will crop them to a nearest size divisible by 256 and then 
#divide all images into patches of 256x256x3. 
image_dataset = []  
for path, subdirs, files in os.walk(root_directory):
    #print(path)  
    dirname = path.split(os.path.sep)[-1]
    if dirname == 'images':   #Find all 'images' directories
        images = os.listdir(path)  #List of all image names in this subdirectory
        for i, image_name in enumerate(images):  
            if image_name.endswith(".jpg"):   #Only read jpg images...
               
                image = cv2.imread(path+"/"+image_name, 1)  #Read each image as BGR
                SIZE_X = (image.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
                SIZE_Y = (image.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
                image = Image.fromarray(image)
                image = image.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
                #image = image.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
                image = np.array(image)             
       
                #Extract patches from each image
                print("Now patchifying image:", path+"/"+image_name)
                patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)  #Step=256 for 256 patches means no overlap
        
                for i in range(patches_img.shape[0]):
                    for j in range(patches_img.shape[1]):
                        
                        single_patch_img = patches_img[i,j,:,:]
                        
                        #Use minmaxscaler instead of just dividing by 255. 
                        single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
                        
                        #single_patch_img = (single_patch_img.astype('float32')) / 255. 
                        single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds.                               
                        image_dataset.append(single_patch_img)
                
  
                
  
 #Now do the same as above for masks
 #For this specific dataset we could have added masks to the above code as masks have extension png
mask_dataset = []  
for path, subdirs, files in os.walk(root_directory):
    #print(path)  
    dirname = path.split(os.path.sep)[-1]
    if dirname == 'masks':   #Find all 'images' directories
        masks = os.listdir(path)  #List of all image names in this subdirectory
        for i, mask_name in enumerate(masks):  
            if mask_name.endswith(".png"):   #Only read png images... (masks in this dataset)
               
                mask = cv2.imread(path+"/"+mask_name, 1)  #Read each image as Grey (or color but remember to map each color to an integer)
                mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
                SIZE_X = (mask.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
                SIZE_Y = (mask.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
                mask = Image.fromarray(mask)
                mask = mask.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
                #mask = mask.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
                mask = np.array(mask)             
       
                #Extract patches from each image
                print("Now patchifying mask:", path+"/"+mask_name)
                patches_mask = patchify(mask, (patch_size, patch_size, 3), step=patch_size)  #Step=256 for 256 patches means no overlap
        
                for i in range(patches_mask.shape[0]):
                    for j in range(patches_mask.shape[1]):
                        
                        single_patch_mask = patches_mask[i,j,:,:]
                        #single_patch_img = (single_patch_img.astype('float32')) / 255. #No need to scale masks, but you can do it if you want
                        single_patch_mask = single_patch_mask[0] #Drop the extra unecessary dimension that patchify adds.                               
                        mask_dataset.append(single_patch_mask) 
 
image_dataset = np.array(image_dataset)
mask_dataset =  np.array(mask_dataset)

#Sanity check, view few mages
import random
import numpy as np
image_number = random.randint(0, len(image_dataset))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(image_dataset[image_number], (patch_size, patch_size, 3)))
plt.subplot(122)
plt.imshow(np.reshape(mask_dataset[image_number], (patch_size, patch_size, 3)))
plt.show()


###########################################################################
"""
RGB to HEX: (Hexadecimel --> base 16)
This number divided by sixteen (integer division; ignoring any remainder) gives 
the first hexadecimal digit (between 0 and F, where the letters A to F represent 
the numbers 10 to 15). The remainder gives the second hexadecimal digit. 
0-9 --> 0-9
10-15 --> A-F

Example: RGB --> R=201, G=, B=

R = 201/16 = 12 with remainder of 9. So hex code for R is C9 (remember C=12)

Calculating RGB from HEX: #3C1098
3C = 3*16 + 12 = 60
10 = 1*16 + 0 = 16
98 = 9*16 + 8 = 152

"""
#Convert HEX to RGB array
# Try the following to understand how python handles hex values...
a=int('3C', 16)  #3C with base 16. Should return 60. 
print(a)
#Do the same for all RGB channels in each hex code to convert to RGB
Building = '#3C1098'.lstrip('#')
Building = np.array(tuple(int(Building[i:i+2], 16) for i in (0, 2, 4))) # 60, 16, 152

Land = '#8429F6'.lstrip('#')
Land = np.array(tuple(int(Land[i:i+2], 16) for i in (0, 2, 4))) #132, 41, 246

Road = '#6EC1E4'.lstrip('#') 
Road = np.array(tuple(int(Road[i:i+2], 16) for i in (0, 2, 4))) #110, 193, 228

Vegetation =  'FEDD3A'.lstrip('#') 
Vegetation = np.array(tuple(int(Vegetation[i:i+2], 16) for i in (0, 2, 4))) #254, 221, 58

Water = 'E2A929'.lstrip('#') 
Water = np.array(tuple(int(Water[i:i+2], 16) for i in (0, 2, 4))) #226, 169, 41

Unlabeled = '#9B9B9B'.lstrip('#') 
Unlabeled = np.array(tuple(int(Unlabeled[i:i+2], 16) for i in (0, 2, 4))) #155, 155, 155

label = single_patch_mask

# Now replace RGB to integer values to be used as labels.
#Find pixels with combination of RGB for the above defined arrays...
#if matches then replace all values in that pixel with a specific integer
def rgb_to_2D_label(label):
    """
    Suply our labale masks as input in RGB format. 
    Replace pixels with specific RGB values ...
    """
    label_seg = np.zeros(label.shape,dtype=np.uint8)
    label_seg [np.all(label == Building,axis=-1)] = 0
    label_seg [np.all(label==Land,axis=-1)] = 1
    label_seg [np.all(label==Road,axis=-1)] = 2
    label_seg [np.all(label==Vegetation,axis=-1)] = 3
    label_seg [np.all(label==Water,axis=-1)] = 4
    label_seg [np.all(label==Unlabeled,axis=-1)] = 5
    
    label_seg = label_seg[:,:,0]  #Just take the first channel, no need for all 3 channels
    
    return label_seg

labels = []
for i in range(mask_dataset.shape[0]):
    label = rgb_to_2D_label(mask_dataset[i])
    labels.append(label)    

labels = np.array(labels)   
labels = np.expand_dims(labels, axis=3)

print("Unique labels in label dataset are: ", np.unique(labels))

#Another Sanity check, view few mages
import random
import numpy as np
image_number = random.randint(0, len(image_dataset))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(image_dataset[image_number])
plt.subplot(122)
plt.imshow(labels[image_number][:,:,0])
plt.show()


############################################################################


n_classes = len(np.unique(labels))
from keras.utils import to_categorical
labels_cat = to_categorical(labels, num_classes=n_classes)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels_cat, test_size = 0.20, random_state = 42)


#######################################
#Parameters for model
# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss
# from sklearn.utils.class_weight import compute_class_weight

# weights = compute_class_weight('balanced', np.unique(np.ravel(labels,order='C')), 
#                               np.ravel(labels,order='C'))
# print(weights)

weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]
dice_loss = sm.losses.DiceLoss(class_weights=weights) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)  #


IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

from simple_multi_unet_model import multi_unet_model, jacard_coef  

metrics=['accuracy', jacard_coef]

def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

model = get_model()
model.compile(optimizer='adam', loss=total_loss, metrics=metrics)
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)
model.summary()

os.makedirs('checkpoints', exist_ok=True)

output_dir = os.path.expanduser('~/Semantic/models')
os.makedirs(output_dir, exist_ok=True)
from keras.models import save_model

checkpoint_path = os.path.join(output_dir, 'best_custom_unet_model.h5')

checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_jacard_coef',
    mode='max',
    save_best_only=True,
    verbose=1,
    save_weights_only=False  # Save full model
)
# ===== AUTO-RESUME LOGIC =====
import json
from keras.models import load_model

resume_training = True
history1 = None
initial_epoch = 0
history_path = os.path.join(output_dir, 'history_custom_unet.json')
final_model_path = os.path.join(output_dir, 'final_custom_unet_model.h5')
interrupted_model_path = os.path.join(output_dir, 'interrupted_custom_unet_model.h5')

if resume_training and os.path.exists(interrupted_model_path):
    print("🔁 Resuming from interrupted model...")
    model = load_model(interrupted_model_path, custom_objects={'jacard_coef': jacard_coef})
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            previous_history = json.load(f)
            initial_epoch = len(previous_history.get('loss', []))  # Resume from where it left
else:
    print("🆕 Starting fresh training...")
from keras.callbacks import Callback

# Custom callback to save model at regular intervals (e.g., every 10 epochs)
class IntervalCheckpoint(Callback):
    def __init__(self, interval, save_path_prefix):
        super(IntervalCheckpoint, self).__init__()
        self.interval = interval
        self.save_path_prefix = save_path_prefix

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            save_path = os.path.join(
                output_dir, f'{self.save_path_prefix}_epoch_{epoch+1}.h5'
            )
            self.model.save(save_path)
            print(f"📌 Model checkpoint saved at: {save_path}")

try:
    print("🔄 Starting training for Custom UNet Model...")

    # Initialize interval saver (every 10 epochs)
    interval_checkpoint = IntervalCheckpoint(interval=10, save_path_prefix='custom_unet')

    # Training starts
    history1 = model.fit(
        X_train, y_train,
        batch_size=16,
        verbose=1,
        epochs=100,
        initial_epoch=initial_epoch,
        validation_data=(X_test, y_test),
        shuffle=False,
        callbacks=[checkpoint, interval_checkpoint]
    )

    # Save final model after training
    final_model_path = os.path.join(output_dir, 'final_custom_unet_model.h5')
    model.save(final_model_path)
    print(f"✅ Final Custom UNet model saved at: {final_model_path}")

except KeyboardInterrupt:
    # Save interrupted model
    interrupted_path = os.path.join(output_dir, 'interrupted_custom_unet_model.h5')
    model.save(interrupted_path)
    print(f"⚠️ Training interrupted! Custom UNet model saved at: {interrupted_path}")

# Save training history
history_dict = history1.history
if os.path.exists(history_path):
    with open(history_path, 'r') as f:
        prev_hist = json.load(f)
    for k in history_dict:
        if k in prev_hist:
            prev_hist[k].extend(history_dict[k])
        else:
            prev_hist[k] = history_dict[k]
    history_dict = prev_hist

with open(history_path, 'w') as f:
    json.dump(history_dict, f)


#Minmaxscaler
#With weights...[0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]   in Dice loss
#With focal loss only, after 100 epochs val jacard is: 0.62  (Mean IoU: 0.6)            
#With dice loss only, after 100 epochs val jacard is: 0.74 (Reached 0.7 in 40 epochs)
#With dice + 5 focal, after 100 epochs val jacard is: 0.711 (Mean IoU: 0.611)
##With dice + 1 focal, after 100 epochs val jacard is: 0.75 (Mean IoU: 0.62)
#Using categorical crossentropy as loss: 0.71

##With calculated weights in Dice loss.    
#With dice loss only, after 100 epochs val jacard is: 0.672 (0.52 iou)


##Standardscaler 
#Using categorical crossentropy as loss: 0.677

#model.save('models/satellite_standard_unet_100epochs_7May2021.hdf5')
############################################################
#TRY ANOTHE MODEL - WITH PRETRINED WEIGHTS
#Resnet backbone
# ============ RESNET34 UNET WITH AUTO-RESUME ============
# ============ RESNET34 UNET WITH AUTO-RESUME ============

BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

X_train_prepr = preprocess_input(X_train)
X_test_prepr = preprocess_input(X_test)

model_resnet_backbone = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=n_classes, activation='softmax')
model_resnet_backbone.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

print(model_resnet_backbone.summary())

# Paths
resnet_checkpoint_path = os.path.join(output_dir, 'best_resnet_model.h5')
resnet_final_model_path = os.path.join(output_dir, 'final_resnet_model.h5')
resnet_interrupted_model_path = os.path.join(output_dir, 'interrupted_resnet_model.h5')
resnet_history_path = os.path.join(output_dir, 'history_resnet_model.json')

# Resume logic
resume_resnet = True
resnet_initial_epoch = 0
resnet_history = None

if resume_resnet and os.path.exists(resnet_interrupted_model_path):
    print("🔁 Resuming ResNet UNet from interrupted model...")
    model_resnet_backbone = load_model(resnet_interrupted_model_path, custom_objects={'jacard_coef': jacard_coef})
    if os.path.exists(resnet_history_path):
        with open(resnet_history_path, 'r') as f:
            previous_resnet_history = json.load(f)
            resnet_initial_epoch = len(previous_resnet_history.get('loss', []))
else:
    print("🆕 Starting ResNet UNet training from scratch...")

# ResNet checkpoint
resnet_checkpoint = ModelCheckpoint(
    filepath=resnet_checkpoint_path,
    monitor='val_jacard_coef',
    mode='max',
    save_best_only=True,
    verbose=1,
    save_weights_only=False
)

try:
    print("🔄 Starting training for ResNet34 UNet Model...")
    history2 = model_resnet_backbone.fit(
        X_train_prepr,
        y_train,
        batch_size=16,
        epochs=100,
        initial_epoch=resnet_initial_epoch,
        verbose=1,
        validation_data=(X_test_prepr, y_test),
        callbacks=[resnet_checkpoint]
    )

    # Save final model
    model_resnet_backbone.save(resnet_final_model_path)
    print(f"✅ Final ResNet model saved at: {resnet_final_model_path}")

    # Save ResNet history only if training wasn't interrupted
    resnet_history_dict = history2.history
    if os.path.exists(resnet_history_path):
        with open(resnet_history_path, 'r') as f:
            prev_hist = json.load(f)
        for k in resnet_history_dict:
            if k in prev_hist:
                prev_hist[k].extend(resnet_history_dict[k])
            else:
                prev_hist[k] = resnet_history_dict[k]
        resnet_history_dict = prev_hist

    with open(resnet_history_path, 'w') as f:
        json.dump(resnet_history_dict, f)

except KeyboardInterrupt:
    # Save interrupted model only (no history written if fit was killed before writing)
    model_resnet_backbone.save(resnet_interrupted_model_path)
    print(f"⚠️ Training interrupted! ResNet model saved at: {resnet_interrupted_model_path}")
    print("ℹ️ Skipping history save because training didn't complete.")


#Minmaxscaler
#With weights...[0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]   in Dice loss
#With focal loss only, after 100 epochs val jacard is:               
#With dice + 5 focal, after 100 epochs val jacard is: 0.73 (reached 0.71 in 40 epochs. So faster training but not better result. )
##With dice + 1 focal, after 100 epochs val jacard is:   
    ##Using categorical crossentropy as loss: 0.755 (100 epochs)
#With calc. weights supplied to model.fit: 
 
#Standard scaler
#Using categorical crossentropy as loss: 0.74
###########################################################
#plot the training and validation accuracy and loss at each epoch
history = history1
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['jacard_coef']
val_acc = history.history['val_jacard_coef']

plt.plot(epochs, acc, 'y', label='Training IoU')
plt.plot(epochs, val_acc, 'r', label='Validation IoU')
plt.title('Training and validation IoU')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()
plt.show()


##################################
from keras.models import load_model
model = load_model("models/satellite_standard_unet_100epochs.hdf5",
                   custom_objects={'dice_loss_plus_2focal_loss': total_loss,
                                   'jacard_coef':jacard_coef})

#IOU
y_pred=model.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=3)
y_test_argmax=np.argmax(y_test, axis=3)


#Using built in keras function for IoU
from keras.metrics import MeanIoU
n_classes = 6
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_test_argmax, y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

# Add after your existing IoU calculation in the training code
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import average_precision_score

def calculate_metrics(y_true, y_pred, n_classes):
    # Flatten arrays
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    metrics = {
        'Overall Accuracy': accuracy_score(y_true_flat, y_pred_flat),
        'Mean Precision': precision_score(y_true_flat, y_pred_flat, average='weighted'),
        'Mean Recall': recall_score(y_true_flat, y_pred_flat, average='weighted'),
        'Mean F1': f1_score(y_true_flat, y_pred_flat, average='weighted'),
        'Class-wise AP': [],
        'Class-wise IoU': []
    }
    
    # Calculate class-wise metrics
    for class_id in range(n_classes):
        # Precision, Recall, F1
        ap = average_precision_score((y_true_flat == class_id).astype(int),
                                    (y_pred_flat == class_id).astype(int))
        intersection = np.sum((y_pred_flat == class_id) & (y_true_flat == class_id))
        union = np.sum((y_pred_flat == class_id) | (y_true_flat == class_id))
        iou = intersection / (union + 1e-7)
        
        metrics['Class-wise AP'].append(ap)
        metrics['Class-wise IoU'].append(iou)
    
    metrics['mAP@0.5'] = np.mean(metrics['Class-wise AP'])
    metrics['Mean IoU'] = np.mean(metrics['Class-wise IoU'])
    
    return metrics

# Add this after your existing IoU calculation
metrics = calculate_metrics(y_test_argmax, y_pred_argmax, n_classes)

print("\n===================== Evaluation Metrics =====================")
print(f"Overall Accuracy: {metrics['Overall Accuracy']:.4f}")
print(f"Mean Precision: {metrics['Mean Precision']:.4f}")
print(f"Mean Recall: {metrics['Mean Recall']:.4f}")
print(f"Mean F1 Score: {metrics['Mean F1']:.4f}")
print(f"Mean IoU: {metrics['Mean IoU']:.4f}")
print(f"mAP@0.5: {metrics['mAP@0.5']:.4f}")

print("\nClass-wise Metrics:")
for idx in range(n_classes):
    print(f"Class {idx} ({LABEL_MAPPING[idx]}):")
    print(f"  AP: {metrics['Class-wise AP'][idx]:.4f}")
    print(f"  IoU: {metrics['Class-wise IoU'][idx]:.4f}")

#######################################################################
#Predict on a few images

import random
test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth=y_test_argmax[test_img_number]
#test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]


plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img)
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth)
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img)
plt.show()

#####################################################################
