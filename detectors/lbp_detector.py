import joblib
import cv2
import numpy as np
from skimage.feature import local_binary_pattern


class LBPDetector:
    def __init__(self, model_path="trained/lbp.joblib"):
        self.model = joblib.load(model_path)

    def predict(self, image):
        img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(img, P=16, R=2, method="uniform")
        histogram = np.histogram(lbp.ravel(), bins=np.arange(0, 10), density=True)[0]
        features = histogram / histogram.sum()
        return self.model.predict(features.reshape(1, -1))[0], lbp
