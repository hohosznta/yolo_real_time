from __future__ import annotations

import argparse
import datetime
from collections.abc import Generator
from typing import TypedDict

import cv2
import numpy as np
from ultralytics import YOLO


class DetectionResult(TypedDict):
    ids: list[int]
    data: list[list[float]]
    frame: cv2.Mat | np.ndarray
    timestamp: float


class LiveStreamDetector:
    """
    A class to perform live stream detection and tracking using YOLO11.
    """

    def __init__(
        self,
        source: str | int = 0,
        model_path: str = '../models/pt/best_yolo11n.pt',
    ):
        """
        Initialise live stream detector with video source, YOLO model path.

        Args:
            source (str | int): The video source - can be:
                - int: Webcam index (0 for default webcam, 1 for second webcam, etc.)
                - str: URL to live video stream or path to video file
            model_path (str): The path to the YOLO11 model file.
        """
        self.source = source
        self.model_path = model_path
        self.model = YOLO(self.model_path)

        # Convert source to int if it's a string representing a number
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        self.cap = cv2.VideoCapture(source)

    def generate_detections(
        self,
    ) -> Generator[
        tuple[list[int], list[list[float]], cv2.Mat | np.ndarray, float],
    ]:
        """
        Yields detection results, timestamp per frame from video capture.

        Yields:
            Generator[Tuple]: Tuple of detection ids, detection data, frame,
            and the current timestamp for each video frame.
        """
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break  # Exit if there is an issue with video capture

            # Get the current timestamp
            now = datetime.datetime.now()
            timestamp = now.timestamp()  # Unix timestamp

            # Run YOLO11 tracking, maintain tracks across frames
            results = self.model.track(source=frame, persist=True)

            # If detections exist, extract IDs and data
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                # Check emptiness, not converting to list
                ids = results[0].boxes.id
                datas = results[0].boxes.data  # Same as above

                # Convert ids and datas to lists if they are not empty
                ids_list = (
                    ids.cpu().numpy().tolist()
                    if ids is not None and len(ids) > 0
                    else []
                )
                datas_list = (
                    datas.cpu().numpy().tolist()
                    if datas is not None and len(datas) > 0
                    else []
                )

                # Yield the results
                yield ids_list, datas_list, frame, timestamp
            else:
                yield [], [], frame, timestamp

            # Exit loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def release_resources(self):
        """
        Releases the video capture object and closes all OpenCV windows.
        """
        self.cap.release()
        cv2.destroyAllWindows()

    def run_detection(self):
        """
        Runs the live stream detection and prints out detection results.
        """
        for ids, datas, frame, timestamp in self.generate_detections():
            print(
                'Timestamp:', datetime.datetime.fromtimestamp(
                    timestamp, tz=datetime.timezone.utc,
                ).strftime('%Y-%m-%d %H:%M:%S'),
            )
            print('IDs:', ids)
            print('Data (xyxy format):')
            print(datas)


def main():
    """
    Main function to run the live stream detection.
    """
    parser = argparse.ArgumentParser(
        description='Perform live stream detection and tracking using YOLO11.',
    )
    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help='Video source: webcam index (0, 1, etc.) or stream URL or video file path (default: 0 for webcam)',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='../models/yolo11n.pt',
        help='Path to the YOLO model',
    )
    args = parser.parse_args()

    # Convert source to int if it's a webcam index
    source = args.source
    if source.isdigit():
        source = int(source)

    # Initialize the detector with the video source and model path
    detector = LiveStreamDetector(source, args.model)

    # Run the detection and print the results
    detector.run_detection()

    # Release resources after detection is complete
    detector.release_resources()


if __name__ == '__main__':
    main()

"""examples
# Use default webcam (index 0)
python main.py

# Use specific webcam (e.g., second camera)
python main.py --source 1
"""