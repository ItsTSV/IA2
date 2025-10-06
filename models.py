import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data


class BasicParkingNet(nn.Module):
    def __init__(self, img_width=40, img_height=60):
        super().__init__()

        # Basic fc network
        self.model = nn.Sequential(
            nn.Linear(img_width * img_height, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Optimizer
        self.optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))
        self.eval()


class CnnParkingNet(nn.Module):
    def __init__(self, in_channels=1, img_height=60, img_width=40):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # blok 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # blok 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_height, img_width)
            n_flatten = self.features(dummy).numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        self.optimizer = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))
        self.eval()