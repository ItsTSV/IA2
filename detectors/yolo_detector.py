from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_path="trained/best_yolo.pt"):
        self.model = YOLO(model_path)
        self.model.eval()

    def detect_all_full(self, image):
        result = self.model([image])[0]

        # Parse much nicer and friendlier format!
        processed = [
            single_detection
            for single_detection in zip(result.boxes.xyxy, result.boxes.conf)
            if single_detection[1] > 0.5
        ]

        return processed
