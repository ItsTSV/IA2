import torch
import torch.nn as nn


class TinyParkingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TinyParkingBlock, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TinyParkingNet(nn.Module):
    def __init__(self):
        super(TinyParkingNet, self).__init__()

        # Extract features
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            TinyParkingBlock(16, 32),
            TinyParkingBlock(32, 64),
            TinyParkingBlock(64, 128),
            TinyParkingBlock(128, 128),
        )

        # Classify stuff
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def load(self, path):
        self.load_state_dict(
            torch.load(
                path,
                weights_only=True,
                map_location="cuda" if torch.cuda.is_available() else "cpu",
            )
        )
        self.eval()
