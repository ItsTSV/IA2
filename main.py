import glob
from utils import *


if __name__ == "__main__":
    # Open map and get coords
    pkm_file = open('parking_map_python.txt', 'r')
    pkm_lines = pkm_file.readlines()
    pkm_coordinates = []
    for line in pkm_lines:
        st_line = line.strip()
        sp_line = list(st_line.split(" "))
        pkm_coordinates.append(sp_line)

    # Create window
    cv2.namedWindow('Fajne parkoviste')

    # Loop through parking lot images
    test_images = [img for img in glob.glob("test_images/*.jpg")]
    test_images.sort()
    for address in test_images:
        # Vars
        predictions = []

        # Open parking img, create copy
        img = cv2.imread(address)
        copy = img.copy()

        # Get individual parking spaces
        for coords in pkm_coordinates:
            # Get single image
            warped = four_point_transform(img, coords)
            warped = cv2.resize(warped, (100, 150))

            # Draw dot at position
            cv2.circle(copy, (int(coords[0]), int(coords[1])), 5, (0, 255, 0), 5)

            # Get prediction (random)
            predictions.append(np.random.choice([0, 1]))

        # Compute F1 Score
        accuracy, f1 = compute_scores(predictions, address.replace(".jpg", ".txt"))
        print(f"Accuracy: {accuracy}")
        print(f"F1: {f1}")

        # Display it
        cv2.imshow('Fajne parkoviste', copy)
        cv2.waitKey(0)
