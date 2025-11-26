import cv2
import numpy as np


class EdgeDetector:
    def __init__(self, threshold1=100, threshold2=200, edge_sum_threshold=1000):
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.edge_sum_threshold = edge_sum_threshold

    def predict(self, image):
        edges = cv2.Canny(image, self.threshold1, self.threshold2)
        edge_sum = np.sum(edges) / 255
        return 1 if edge_sum > self.edge_sum_threshold else 0, edges
