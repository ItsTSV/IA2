import cv2
import numpy as np
import joblib
from skimage.feature import local_binary_pattern, hog, haar_like_feature
from skimage.transform import integral_image
import skimage as ski
import matplotlib.pyplot as plt
import glob
from models import BasicParkingNet, CnnParkingNet, ParkingMobileNetV3
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

    def predict(self, image):
        img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (40, 60))
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            output = self.network(img_tensor)
        prediction = 1 if output.item() > 0.5 else 0
        return prediction


class CnnDetector:
    def __init__(self, model_path="models/cnn_parking_net.pth"):
        self.network = CnnParkingNet()
        self.network.load(model_path)

    def predict(self, image):
        img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (32, 64))
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        img_tensor /= 255.0
        with torch.no_grad():
            output = self.network(img_tensor)
        prediction = 1 if output.item() > 0.5 else 0
        return prediction


class MobileNetDetector:
    def __init__(self, model_path="models/parking_mobilenet.pth"):
        self.network = ParkingMobileNetV3()
        self.network.load(model_path)
        self.network.eval()

    def predict(self, image):
        img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (224, 224))
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            output = self.network(img_tensor)
        prediction = 1 if output.item() > 0.5 else 0
        return prediction