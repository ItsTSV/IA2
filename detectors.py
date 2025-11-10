import cv2
import numpy as np
import joblib
import torchvision.models.detection
from skimage.feature import local_binary_pattern, hog, haar_like_feature
from skimage.transform import integral_image
import skimage as ski
import matplotlib.pyplot as plt
import glob

from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)

from models import (
    BasicParkingNet,
    CnnParkingNet,
    ParkingMobileNetV3,
    ParkingEfficientNet,
)
import torch


class EdgeDetector:
    def __init__(self, threshold1=100, threshold2=200, edge_sum_threshold=1000):
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.edge_sum_threshold = edge_sum_threshold

    def predict(self, image):
        edges = cv2.Canny(image, self.threshold1, self.threshold2)
        edge_sum = np.sum(edges) / 255
        return 1 if edge_sum > self.edge_sum_threshold else 0, edges


class LBPDetector:
    def __init__(self, model_path="models/lbp.joblib"):
        self.model = joblib.load(model_path)

    def predict(self, image):
        img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(img, P=16, R=2, method="uniform")
        histogram = np.histogram(lbp.ravel(), bins=np.arange(0, 10), density=True)[0]
        features = histogram / histogram.sum()
        return self.model.predict(features.reshape(1, -1))[0], lbp


class HOGDetector:
    def __init__(self, model_path="models/hog.joblib"):
        self.model = joblib.load(model_path)

    def predict(self, image):
        img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        features, img = hog(
            img,
            orientations=8,
            pixels_per_cell=(16, 16),
            cells_per_block=(1, 1),
            visualize=True,
        )
        return self.model.predict(features.reshape(1, -1))[0], img


class HaarDetector:
    def __init__(self, model_path="models/haar.joblib"):
        self.model = joblib.load(model_path)

    def predict(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (30, 45))
        ii = integral_image(img)
        features = haar_like_feature(
            ii, 0, 0, ii.shape[0], ii.shape[1], feature_type=["type-2-x", "type-2-y"]
        )
        return self.model.predict(features.reshape(1, -1))[0], img

    def plot_best_features(self, images_path="train_images/full/*.png"):
        """Plots best features used by the Haar classifier."""
        images = [
            cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in glob.glob(images_path)
        ]
        images = [cv2.resize(img, (30, 45)) for img in images]
        images = np.array(images)

        feature_coord, feature_type = ski.feature.haar_like_feature_coord(
            width=30, height=45, feature_type=["type-2-x", "type-2-y"]
        )

        best_features = np.argsort(self.model.feature_importances_)[::-1]
        fig, axes = plt.subplots(3, 2)
        for idx, ax in enumerate(axes.ravel()):
            image = images[0]
            image = ski.feature.draw_haar_like_feature(
                image,
                0,
                0,
                images.shape[2],
                images.shape[1],
                [feature_coord[best_features[idx]]],
            )
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                f"Feature importance: {self.model.feature_importances_[best_features[idx]]:.4f}"
            )

        plt.tight_layout()
        plt.show()


class BasicNeuralDetector:
    def __init__(self, model_path="models/basic_parking_net.pth"):
        self.network = BasicParkingNet()
        self.network.load(model_path)

    def predict(self, image, confidence_adjustment: float = 0):
        img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (40, 60))
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            output = self.network(img_tensor)
        prediction = 1 if output.item() > 0.5 - confidence_adjustment else 0
        return prediction


class CnnDetector:
    def __init__(self, model_path="models/cnn_parking_net.pth"):
        self.network = CnnParkingNet()
        self.network.load(model_path)

    def predict(self, image, confidence_adjustment: float = 0):
        img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (32, 64))
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        img_tensor /= 255.0
        with torch.no_grad():
            output = self.network(img_tensor)
        prediction = 1 if output.item() > 0.5 - confidence_adjustment else 0
        return prediction


class MobileNetDetector:
    def __init__(self, model_path="models/parking_mobilenet.pth"):
        self.network = ParkingMobileNetV3()
        self.network.load(model_path)
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


class EfficientNetDetector:
    def __init__(self, model_path="models/parking_efficientnet.pth"):
        self.network = ParkingEfficientNet()
        self.network.load(model_path)
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


class FasterRCNNDetector:
    def __init__(self):
        self.model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        )
        self.model.eval()
        self.allowed_indices = [8, 3, 6, 4, 77, 7, 33]

    def detect_all_full(self, image):
        img_tensor = (
            torch.tensor(image, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
        )
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
            if single_detection[1] in self.allowed_indices
            and single_detection[2] > 0.7
        ]

        return processed
