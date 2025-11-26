import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class ParkingMobileNetV3(nn.Module):
    def __init__(self):
        super(ParkingMobileNetV3, self).__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT
        self.base = mobilenet_v3_small(weights=weights)

        # Turn off gradients
        for param in self.base.features.parameters():
            param.requires_grad = False

        # Turn on gradients for last conv blocks and fc layers
        for name, module in list(self.base.features.named_children())[14:]:
            for param in module.parameters():
                param.requires_grad = True

        # Replace MobileNet head with head for my classification
        in_features = self.base.classifier[0].in_features
        self.base.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.base(x)

    def load(self, path):
        self.load_state_dict(
            torch.load(
                path,
                weights_only=True,
                map_location="cuda" if torch.cuda.is_available() else "cpu",
            )
        )
        self.eval()
