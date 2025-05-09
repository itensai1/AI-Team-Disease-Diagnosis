from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights




class EfficientNetModel(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetModel, self).__init__()
        self.base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.base_model.classifier[1] = nn.Linear(
            self.base_model.classifier[1].in_features, num_classes
        )
    
    def forward(self, x):
        return self.base_model(x)