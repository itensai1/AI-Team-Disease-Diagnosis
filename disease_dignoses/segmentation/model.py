import json
import os
from typing import Dict, Any, Optional
import torch
import torch.nn.functional as F
from torch import nn
from .segmentor_arch import SegmentorArch
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import logging
import traceback

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
config_path = os.path.join(project_root, "config.json")

with open(config_path, "r") as f:
    config = json.load(f)

class Model:
    _instance = None

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Resolve the model weights path
        weights_path = os.path.join(project_root, config["PRE_TRAINED_MODEL_SEGMENTOR"])
        
        segmentor = SegmentorArch(config["NUM_CLASSES"]).to(self.device)
        state_dict = torch.load(weights_path, map_location=self.device)
        segmentor.load_state_dict(state_dict)
        segmentor = segmentor.eval()
        self.segmentor = segmentor.to(self.device)

    def preprocess(self, image):
        transforms_segmenation = A.Compose([
            A.Resize(config["IMG_SIZE"], config["IMG_SIZE"]),
            ToTensorV2()
        ])
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transformed = transforms_segmenation(image=img)
        tensor = transformed["image"].unsqueeze(0).float().to(self.device)  
        return tensor

    def predict(self, image):
        image = self.preprocess(image)

        with torch.no_grad():
            output = self.segmentor(image)
            output = torch.sigmoid(output)
            masks = (output > 0.3).float()
            masks = masks.squeeze().cpu().numpy()
            masks = masks.transpose(1, 2, 0)
            masks = (masks * 255).astype("uint8")

        return masks

model = Model()
def get_segmentor_model():
    return model

    
    
