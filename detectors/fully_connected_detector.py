import cv2
import torch
from models.medium_fully_connected_net import FullyConnectedNetMedium
from models.small_fully_connected_net import FullyConnectedNetSmall
from models.big_fully_connected_net import FullyConnectedNetBig


class FullyConnectedDetector:
    def __init__(self, model):
        if model == "small":
            self.network = FullyConnectedNetSmall()
            self.network.load("trained/small_fully_connected_net.pth")
        elif model == "medium":
            self.network = FullyConnectedNetMedium()
            self.network.load("trained/medium_fully_connected_net.pth")
        elif model == "big":
            self.network = FullyConnectedNetBig()
            self.network.load("trained/big_fully_connected_net.pth")
        else:
            raise ValueError("Model must be 'small', 'medium' or 'big'")
    def predict(self, image, confidence_adjustment: float = 0):
        img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (40, 60))
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            output = self.network(img_tensor)
        prediction = 1 if output.item() > 0.5 - confidence_adjustment else 0
        return prediction
