import numpy as np
import cv2


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, one_c):
    """
    Returns:
        x,y top left
        x,y top right
        x,y bottom left
        x,y bottom right
    """
    pts = [((float(one_c[0])), float(one_c[1])),
           ((float(one_c[2])), float(one_c[3])),
           ((float(one_c[4])), float(one_c[5])),
           ((float(one_c[6])), float(one_c[7]))]

    rect = order_points(np.array(pts))
    (tl, tr, br, bl) = rect
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped


def compute_scores(predicted: list, real_path: str) -> tuple:
    """Calculates real vs. predicted scores

    Returns:
        Accuracy, F1
    """
    with open(real_path, "r") as file:
        true_values = [1 if line.strip() == "1" else 0 for line in file.readlines()]

    if len(true_values) != len(predicted):
        print("Prediction size mismatch!")

    tp, fp, tn, fn = 0, 0, 0, 0
    for pred, real in zip(predicted, true_values):
        if real == 1 and pred == 1:
            tp += 1
        elif real == 0 and pred == 1:
            fp += 1
        elif real == 0 and pred == 0:
            tn += 1
        elif real == 1 and pred == 0:
            fn += 1

    accuracy = (tp + fp) / (tp + fp + tn + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)

    return accuracy, f1
