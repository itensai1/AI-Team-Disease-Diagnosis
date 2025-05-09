import json

import torch
import torch.nn.functional as F
from torch import nn
from .disease_classifer import EfficientNetModel
from PIL import Image
from torchvision import transforms
import pickle

with open("config.json", "r") as json_file:
    config = json.load(json_file)

idx_to_name = pickle.load(open('assets/idx_to_name.pkl', 'rb'))

class Model:
    _instance = None

    def __init__(self):
        if Model._instance is not None:
            raise Exception("This class is a singleton!")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            classifier = EfficientNetModel(config["num_classes"]).to(self.device)
            classifier.load_state_dict(torch.load(config["PRE_TRAINED_MODEL_CLASSIFIER"]))
            classifier = classifier.eval()
            self.classifier = classifier.to(self.device)
        except Exception as e:
            raise Exception(f"Failed to initialize model: {str(e)}")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def transform_image(self, image):
        transforms_classify = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
        ])
        return transforms_classify(image)

    def predict(self, image):
        image = self.transform_image(image)
        image = image.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.classifier(image)
            output = F.softmax(output, dim=1)
            _, idx = torch.max(output, dim=1)
            label = idx_to_name[idx.item()]
            confidence = torch.max(output, axis=1).values.item()
        return {"label": label, "confidence": confidence}
    
    


def get_model():
    return Model.get_instance()
