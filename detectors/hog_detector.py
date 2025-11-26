import joblib
import cv2
from skimage.feature import hog


class HOGDetector:
    def __init__(self, model_path="trained/hog.joblib"):
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
