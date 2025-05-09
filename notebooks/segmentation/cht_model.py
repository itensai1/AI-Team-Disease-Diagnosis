import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import os
from glob import glob

# Configurations
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
EPOCHS = 20
LR = 1e-4
IMG_SIZE = 256
NUM_CLASSES = 5  # Optic Disc, Microaneurysms, Hemorrhages, Soft Exudates, Hard Exudates

# Dataset Class
class IDRiDDataset(Dataset):
    def _init_(self, image_paths, mask_dirs, transform=None):
        self.image_paths = image_paths
        self.mask_dirs = mask_dirs  # List of 5 mask directories
        self.transform = transform
    
    def _len_(self):
        return len(self.image_paths)
    
    def _getitem_(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Ensure consistent image size
        
        # Load multi-class masks
        mask = np.zeros((NUM_CLASSES, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        image_name = os.path.basename(self.image_paths[idx]).split('.')[0]
        
        for class_idx, mask_dir in enumerate(self.mask_dirs):
            mask_path = os.path.join(mask_dir, f"{image_name}.tif")  # Updated to handle .tif masks
            if os.path.exists(mask_path):
                class_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if class_mask is not None:
                    class_mask = cv2.resize(class_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
                    mask[class_idx] = class_mask / 255.0  # Normalize to [0,1]
        
        # if self.transform:
        #     augmented = self.transform(image=img, mask=mask.transpose(1, 2, 0))  # Move mask channels to last dim
        #     img, mask = augmented['image'], augmented['mask'].permute(2, 0, 1)  # Restore channel-first format
        
        return img, mask

# Data Augmentations
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    ToTensorV2()
])

test_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    ToTensorV2()
])

# Load Dataset Paths
image_dir = "path_to_images"
mask_dirs = [
    "path_to_masks/optic_disc",
    "path_to_masks/microaneurysms",
    "path_to_masks/hemorrhages",
    "path_to_masks/soft_exudates",
    "path_to_masks/hard_exudates"
]

image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))

# Keep only images that have at least one corresponding mask
valid_image_names = set()
for mask_dir in mask_dirs:
    mask_files = glob(os.path.join(mask_dir, "*.tif"))
    valid_image_names.update(os.path.basename(f).split('.')[0] for f in mask_files)

image_paths = [os.path.join(image_dir, f"{name}.jpg") for name in valid_image_names if os.path.exists(os.path.join(image_dir, f"{name}.jpg"))]

# Split into train & test
train_size = int(0.8 * len(image_paths))
train_imgs, test_imgs = image_paths[:train_size], image_paths[train_size:]

train_dataset = IDRiDDataset(train_imgs, mask_dirs, transform=train_transform)
test_dataset = IDRiDDataset(test_imgs, mask_dirs, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model - U-Net++ with ResNet34 Encoder
model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
    activation=None
).to(DEVICE)

# Loss & Optimizer
loss_fn = smp.losses.DiceLoss(mode='multiclass') + nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training Loop
def train():
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss/len(train_loader):.4f}")

# Evaluation Function
def evaluate():
    model.eval()
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            outputs = torch.softmax(model(imgs), dim=1)
            preds = torch.argmax(outputs, dim=1)  # Get predicted class per pixel
            # TODO: Add evaluation metrics (Dice Score, IoU, etc.)

# Run Training
train()
evaluate()