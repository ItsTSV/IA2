import torch
import torch.nn as nn
import torch.optim as optim


class FullyConnectedNet(nn.Module):
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
        self.load_state_dict(
            torch.load(
                path,
                weights_only=True,
                map_location="cuda" if torch.cuda.is_available() else "cpu",
            )
        )
        self.eval()
