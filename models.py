import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torchvision.models.mobilenetv3 import *

class BasicParkingNet(nn.Module):
    def __init__(self, img_width=40, img_height=60):
        super().__init__()

        # Basic fc network
        self.model = nn.Sequential(
            nn.Linear(img_width * img_height, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Optimizer
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))
        self.eval()


class CnnParkingNet(nn.Module):
    def __init__(self, in_channels=1, img_height=64, img_width=32):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=3),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_height, img_width)
            n_flatten = self.features(dummy).numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, 256),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.optimizer = optim.AdamW(self.parameters(), lr=1e-5)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))
        self.eval()


class ParkingMobileNetV3(nn.Module):
    def __init__(self):
        super(ParkingMobileNetV3, self).__init__()
        weights = MobileNet_V3_Large_Weights.DEFAULT
        self.base = mobilenet_v3_large(weights=weights)

        # Turn off gradients
        for param in self.base.features.parameters():
            param.requires_grad = False

        # Turn on gradients for last conv blocks and fc layers
        for name, module in list(self.base.features.named_children())[14:]:
            for param in module.parameters():
                param.requires_grad = True

        # Replace first conv layer to accept 1 channel input
        first_conv = self.base.features[0][0]
        self.base.features[0][0] = nn.Conv2d(1, first_conv.out_channels, kernel_size=first_conv.kernel_size,
                                             stride=first_conv.stride, padding=first_conv.padding, bias=first_conv.bias)

        # Replace MobileNet head with head for my classification
        in_features = self.base.classifier[0].in_features
        self.base.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

        # Optimizer
        self.optimizer = optim.AdamW(self.parameters(), lr=1e-4)

    def forward(self, x):
        return self.base(x)

    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))
        self.eval()