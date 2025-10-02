import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data


class BasicParkingNet(nn.Module):
    def __init__(self, img_width=40, img_height=60):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_width * img_height, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.optimizer = optim.AdamW(self.model.parameters())

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()
