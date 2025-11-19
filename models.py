import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torchvision.models.mobilenetv3 import *
from torchvision.models.efficientnet import *


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
        self.load_state_dict(torch.load(path, weights_only=True, map_location="cuda" if torch.cuda.is_available() else "cpu"))
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
        self.load_state_dict(torch.load(path, weights_only=True, map_location="cuda" if torch.cuda.is_available() else "cpu"))
        self.eval()


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
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.base(x)

    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True, map_location="cuda" if torch.cuda.is_available() else "cpu"))
        self.eval()


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
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.base(x)

    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True, map_location="cuda" if torch.cuda.is_available() else "cpu"))
        self.eval()


class MyVisionTransformer(nn.Module):
    def __init__(self, patch_size=16, embedding_length=768, img_size=224, layer_count=3):
        super(MyVisionTransformer, self).__init__()

        # Parameters
        self.patch_size = patch_size
        self.embedding_length = embedding_length
        self.patch_count = (img_size // patch_size)**2

        # Split img into patches
        self.patch_layer = nn.Conv2d(3, embedding_length, kernel_size=patch_size,
                                     stride=patch_size)

        # CLS token + Positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_length))
        self.positional_embedding = nn.Parameter(torch.randn(1, self.patch_count + 1, embedding_length))

        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_length,
            nhead=12,
            dim_feedforward=embedding_length * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layer_count)

        # Classification head
        self.classification_head = nn.Sequential(
            nn.LayerNorm(embedding_length),
            nn.Linear(embedding_length, 1)
        )

    def forward(self, x):
        # Get patches
        x = self.patch_layer(x)
        x = x.flatten(start_dim=2, end_dim=-1).transpose(1, 2)

        # Token + embeddings
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.positional_embedding

        # Transformer
        x = self.transformer(x)

        # Classify
        x = x[:, 0, :]
        predicted = self.classification_head(x)

        return predicted

    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True, map_location="cuda" if torch.cuda.is_available() else "cpu"))
        self.eval()


