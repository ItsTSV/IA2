import cv2
import torch
from models.fully_connected_net import FullyConnectedNet


class FullyConnectedDetector:
    def __init__(self, model_path="trained/basic_parking_net.pth"):
        self.network = FullyConnectedNet()
        self.network.load(model_path)

    def predict(self, image, confidence_adjustment: float = 0):
        img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (40, 60))
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            output = self.network(img_tensor)
        prediction = 1 if output.item() > 0.5 - confidence_adjustment else 0
        return prediction
