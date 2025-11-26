import glob
from utils import *
from detectors import *


# What models to use?
feature_based = 0
my_fc = 0
big_parking_net = 0
tiny_parking_net = 0
mobilenet = 0
efficientnet = 1
my_vit = 0

# How will two stage detection be used?
use_two_stage = 0
show_two_stage = 0
two_stage_model = "faster"
two_stage_penalty = 1

# YOLO
use_yolo = 0
show_yolo = 0
yolo_penalty = 1

if __name__ == "__main__":
    # Open map and get coords
    pkm_file = open("test_images/parking_map_python.txt", "r")
    pkm_lines = pkm_file.readlines()
    pkm_coordinates = []
    for line in pkm_lines:
        st_line = line.strip()
        sp_line = list(st_line.split(" "))
        pkm_coordinates.append(sp_line)

    # Create window
    cv2.namedWindow("Fajne parkoviste")
    if feature_based:
        cv2.namedWindow("Edges")
        cv2.namedWindow("LBP")
        cv2.namedWindow("HOG")
        cv2.namedWindow("HAAR")

    # Create detectors
    edge_detector = EdgeDetector()
    lbp_detector = LBPDetector()
    hog_detector = HOGDetector()
    haar_detector = HaarDetector()
    basic_neural_detector = FullyConnectedDetector()
    big_parking_detector = NeuralDetector("big_parking_net")
    tiny_parking_detector = NeuralDetector("tiny_parking_net")
    mobilenet_neural_detector = NeuralDetector("mobilenet")
    efficientnet_neural_detector = NeuralDetector("efficientnet")
    two_stage_detector = TwoStageDetector(two_stage_model)
    yolo_detector = YOLODetector()
    vit_detector = NeuralDetector("vit")

    # Prepare all predictions
    all_predictions = []

    # Loop through parking lot images
    test_images = [img for img in glob.glob("test_images/*.jpg")]
    test_images.sort()
    for address in test_images:
        # Vars
        predictions = []

        # Open parking img, create copy
        img = cv2.imread(address)
        copy = img.copy()

        # Detect parking spaces with Faster R-CNN
        if use_two_stage:
            detected_two_stage = two_stage_detector.detect_all_full(img)
            if show_two_stage:
                for det in detected_two_stage:
                    cv2.rectangle(copy, (int(det[0][0]), int(det[0][1])), (int(det[0][2]), int(det[0][3])), (255, 0, 0), 2)

        # Detect parking spaces with YOLO
        if use_yolo:
            detected_yolo = yolo_detector.detect_all_full(img)
            if show_yolo:
                for det in detected_yolo:
                    cv2.rectangle(copy, (int(det[0][0]), int(det[0][1])), (int(det[0][2]), int(det[0][3])), (255, 0, 0), 2)

        # Get individual parking spaces
        for coords in pkm_coordinates:
            # Get single image
            warped = four_point_transform(img, coords)
            warped = cv2.resize(warped, (100, 150))

            # Round predictions
            round_predictions = []

            # If Faster or YOLO is used, adjust confidence based on detections
            adjustment = 0
            if use_two_stage:
                if any(intersects(coords, det[0]) for det in detected_two_stage):
                    adjustment += two_stage_penalty

            if use_yolo:
                if any(intersects(coords, det[0]) for det in detected_yolo):
                    adjustment += yolo_penalty

            # Get predictions
            if feature_based:
                edge_prediction, edges = edge_detector.predict(warped)
                lbp_prediction, histogram = lbp_detector.predict(warped)
                hog_prediction, hog_img = hog_detector.predict(warped)
                haar_prediction, haar_img = haar_detector.predict(warped)
                round_predictions.extend([edge_prediction, lbp_prediction, hog_prediction, haar_prediction])

            if my_fc:
                prediction = basic_neural_detector.predict(warped, adjustment)
                round_predictions.append(prediction)

            if big_parking_net:
                prediction = big_parking_detector.predict(warped, adjustment)
                round_predictions.append(prediction)

            if tiny_parking_net:
                prediction = tiny_parking_detector.predict(warped, adjustment)
                round_predictions.append(prediction)

            if mobilenet:
                prediction = mobilenet_neural_detector.predict(warped, adjustment)
                round_predictions.append(prediction)

            if efficientnet:
                prediction = efficientnet_neural_detector.predict(warped, adjustment)
                round_predictions.append(prediction)

            if my_vit:
                prediction = vit_detector.predict(warped)
                round_predictions.append(prediction)

            prediction = np.argmax(np.bincount(round_predictions))
            predictions.append(prediction)

            # Draw circle
            if prediction:
                draw_bounds(copy, coords, (0, 255, 0))
            else:
                draw_bounds(copy, coords, (0, 0, 255))

            # Display features
            if feature_based:
                cv2.imshow("Edges", edges)
                cv2.imshow("LBP", histogram)
                cv2.imshow("HOG", hog_img)
                cv2.imshow("HAAR", haar_img)

            cv2.imshow("Fajne parkoviste", copy)
            cv2.waitKey(1)

        # Compute scores from all methods
        accuracy, f1 = compute_scores(predictions, address.replace(".jpg", ".txt"))
        all_predictions.append(accuracy)
        predictions.clear()

        # Show round stats
        print(f"-----\n{address}\nAccuracy: {accuracy}\nF1: {f1}\n-----\n")

        # Display it
        cv2.imshow("Fajne parkoviste", copy)
        cv2.waitKey(1)

    # Compute all scores
    accuracy_all = sum(all_predictions) / len(all_predictions)
    print("=================================")
    print(f"All images - Accuracy: {accuracy_all}")
    print("=================================")

    # Cleanup
    cv2.destroyAllWindows()
