import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class ParkingEfficientNet(nn.Module):
    def __init__(self):
        super(ParkingEfficientNet, self).__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        self.base = efficientnet_b0(weights=weights)

        # Turn off gradients
        for param in self.base.features.parameters():
            param.requires_grad = False

        # Turn on gradients for last conv blocks and fc layers
        for name, module in list(self.base.features.named_children())[-4:]:
            for param in module.parameters():
                param.requires_grad = True

        # Replace EfficientNet head with head for my classification
        in_features = self.base.classifier[1].in_features
        self.base.classifier = nn.Sequential(
            nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1)
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
