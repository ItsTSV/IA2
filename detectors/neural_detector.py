import cv2
import torch
from models import ParkingMobileNetV3, ParkingEfficientNet, TinyParkingNet, BigParkingNet, MyVisionTransformer


class NeuralDetector:
    def __init__(self, model):
        if model == "mobilenet":
            self.network = ParkingMobileNetV3()
            self.network.load("trained/parking_mobilenet.pth")
        elif model == "efficientnet":
            self.network = ParkingEfficientNet()
            self.network.load("trained/parking_efficientnet.pth")
        elif model == "big_parking_net":
            self.network = BigParkingNet()
            self.network.load("trained/big_parking_net.pth")
        elif model == "tiny_parking_net":
            self.network = TinyParkingNet()
            self.network.load("trained/tiny_parking_net.pth")
        elif model == "vit":
            self.network = MyVisionTransformer()
            self.network.load("trained/parking_vit.pth")
        else:
            raise ValueError(f"Unknown model: {model}")
        self.network.eval()

    def predict(self, image, confidence_adjustment: float = 0):
        img = cv2.resize(image, (224, 224))
        img_tensor = (
            torch.tensor(img, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
        )
        img_tensor /= 255.0
        with torch.no_grad():
            output = self.network(img_tensor)
        prediction = 1 if output.item() > 0.5 - confidence_adjustment else 0
        return prediction
