import torch
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
)


class TwoStageDetector:
    def __init__(self, model="faster"):
        self.model = (
            fasterrcnn_mobilenet_v3_large_fpn(
                weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            )
            if model == "faster"
            else ssdlite320_mobilenet_v3_large(
                weights=SSDLite320_MobileNet_V3_Large_Weights
            )
        )
        self.model.eval()
        self.allowed_indices = [
            2, 3, 4, 5, 6, 7, 8, 9,     # Vehicles
            10, 11, 13, 14, 15,         # Misc
            18, 19, 21,                 # Animals
            33, 62, 72, 77, 82, 84      # Box like objects
        ]

    def detect_all_full(self, image):
        img_tensor = (torch.tensor(image, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2))
        img_tensor /= 255.0
        with torch.no_grad():
            outputs = self.model(img_tensor)

        # Parse the absolutely terrible list(map{tensor}) structure
        processed = [
            single_detection
            for single_detection in zip(
                outputs[0]["boxes"].detach().cpu().numpy(),
                outputs[0]["labels"].detach().cpu().numpy(),
                outputs[0]["scores"].detach().cpu().numpy(),
            )
            if single_detection[1] in self.allowed_indices and single_detection[2] > 0.5
        ]

        return processed
