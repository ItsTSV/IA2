import glob
from utils import *
from detectors import *
import seaborn as sns
import matplotlib.pyplot as plt


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
    cv2.namedWindow("Edges")
    cv2.namedWindow("LBP")
    cv2.namedWindow("HOG")
    cv2.namedWindow("HAAR")

    # Create detectors
    edge_detector = EdgeDetector()
    lbp_detector = LBPDetector()
    hog_detector = HOGDetector()
    haar_detector = HaarDetector()
    basic_neural_detector = BasicNeuralDetector()

    # Visualize most important Haar features
    haar_features = haar_detector.plot_best_features()

    # Loop through parking lot images
    test_images = [img for img in glob.glob("test_images/*.jpg")]
    test_images.sort()
    for address in test_images:
        # Vars
        predictions = []
        predictions_edge = []
        predictions_lbp = []
        predictions_hog = []
        predictions_haar = []

        # Open parking img, create copy
        img = cv2.imread(address)
        copy = img.copy()

        # Get individual parking spaces
        for coords in pkm_coordinates:
            # Get single image
            warped = four_point_transform(img, coords)
            warped = cv2.resize(warped, (100, 150))

            # Get predictions
            '''
            edge_prediction, edges = edge_detector.predict(warped)
            lbp_prediction, histogram = lbp_detector.predict(warped)
            hog_prediction, hog_img = hog_detector.predict(warped)
            haar_prediction, haar_img = haar_detector.predict(warped)

            # Calculate final prediction
            predictions_array = np.array(
                [edge_prediction, lbp_prediction, hog_prediction, haar_prediction]
            )
            prediction = np.argmax(np.bincount(predictions_array))

            # Append predictions
            predictions.append(prediction)
            predictions_edge.append(edge_prediction)
            predictions_lbp.append(lbp_prediction)
            predictions_hog.append(hog_prediction)
            predictions_haar.append(haar_prediction)
            '''

            prediction = basic_neural_detector.predict(warped)
            predictions.append(prediction)

            # Draw circle
            if prediction:
                draw_bounds(copy, coords, (0, 255, 0))
            else:
                draw_bounds(copy, coords, (0, 0, 255))

            # Display features
            '''
            cv2.imshow("Edges", edges)
            cv2.imshow("LBP", histogram)
            cv2.imshow("HOG", hog_img)
            cv2.imshow("HAAR", haar_img)
            '''
            cv2.imshow("Fajne parkoviste", copy)
            cv2.waitKey(10)

        # Compute scores from all methods
        accuracy, f1 = compute_scores(predictions, address.replace(".jpg", ".txt"))
        '''
        accuracy_edge, f1_edge = compute_scores(predictions_edge, address.replace(".jpg", ".txt"))
        accuracy_lbp, f1_lbp = compute_scores(predictions_lbp, address.replace(".jpg", ".txt"))
        accuracy_hog, f1_hog = compute_scores(predictions_hog, address.replace(".jpg", ".txt"))
        accuracy_haar, f1_haar = compute_scores(predictions_haar, address.replace(".jpg", ".txt"))
        '''
        print("---------------------------------")
        print(f"{address}")
        print(f"Accuracy: {accuracy}, F1: {f1}")
        '''
        print(f"Edge Accuracy: {accuracy_edge}, F1: {f1_edge}")
        print(f"LBP Accuracy: {accuracy_lbp}, F1: {f1_lbp}")
        print(f"HOG Accuracy: {accuracy_hog}, F1: {f1_hog}")
        print(f"HAAR Accuracy: {accuracy_haar}, F1: {f1_haar}")
        '''
        print("---------------------------------")

        # Charts
        '''
        sns.barplot(
            x=["Edge", "LBP", "HOG", "HAAR", "Final"],
            y=[accuracy_edge, accuracy_lbp, accuracy_hog, accuracy_haar, accuracy],
        )
        plt.show()
        '''

        # Display it
        cv2.imshow("Fajne parkoviste", copy)
        cv2.waitKey(0)
