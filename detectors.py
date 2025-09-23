import cv2
import numpy as np
import joblib
from skimage.feature import local_binary_pattern, hog, haar_like_feature
from skimage.transform import integral_image


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
        features = haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1],
                                     feature_type=['type-2-x', 'type-2-y'])
        return self.model.predict(features.reshape(1, -1))[0], img
