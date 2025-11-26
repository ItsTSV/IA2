import joblib
import cv2
from skimage.feature import haar_like_feature
from skimage.transform import integral_image


class HaarDetector:
    def __init__(self, model_path="trained/haar.joblib"):
        self.model = joblib.load(model_path)

    def predict(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (30, 45))
        ii = integral_image(img)
        features = haar_like_feature(
            ii, 0, 0, ii.shape[0], ii.shape[1], feature_type=["type-2-x", "type-2-y"]
        )
        return self.model.predict(features.reshape(1, -1))[0], img
