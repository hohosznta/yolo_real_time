from __future__ import annotations

import argparse
import asyncio
import datetime
import gc
from collections.abc import AsyncGenerator
from typing import TypedDict

import cv2
import numpy as np


class InputData(TypedDict):
    source: int
    capture_interval: int


class ResultData(TypedDict):
    frame: np.ndarray
    timestamp: float


class StreamCapture:
    """
    A class to capture frames from a video stream or webcam.
    """

    def __init__(self, source: str | int = 0, capture_interval: int = 15):
        """
        Initialises the StreamCapture with the given video source.

        Args:
            source (str | int): Video source - can be:
                - int: Webcam index (0 for default webcam, 1 for second webcam, etc.)
            capture_interval (int, optional): The interval at which frames
                should be captured. Defaults to 15.
        """
        # Video source (webcam index)
        self.source = source
        # Video capture object
        self.cap: cv2.VideoCapture | None = None
        # Frame capture interval in seconds
        self.capture_interval = capture_interval
        # Flag to indicate successful capture
        self.successfully_captured = False

    async def initialise_stream(self, source: str | int | None = None) -> None:
        """
        Initialises the video stream or webcam.

        Args:
            source (int | None): The video source. If None, uses self.source.
        """
        if source is None:
            source = self.source

        # Convert string number to int for webcam index
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Set additional properties for webcams
        if isinstance(source, int):
            # Webcam specific settings
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            await asyncio.sleep(2)
            self.cap.open(source)

    async def release_resources(self) -> None:
        """
        Releases resources like the capture object.
        """
        if self.cap:
            self.cap.release()
            self.cap = None
        gc.collect()

    async def execute_capture(
        self,
    ) -> AsyncGenerator[tuple[np.ndarray, float]]:
        """
        Captures frames from the video source and yields them with timestamps.

        Yields:
            Tuple[np.ndarray, float]: The captured frame and the timestamp.
        """
        await self.initialise_stream()
        last_process_time = datetime.datetime.now() - datetime.timedelta(
            seconds=self.capture_interval,
        )
        fail_count = 0  # Counter for consecutive failures

        while True:
            if self.cap is None:
                await self.initialise_stream()

            ret, frame = (
                self.cap.read() if self.cap is not None else (False, None)
            )

            if not ret or frame is None:
                fail_count += 1
                print(
                    'Failed to read frame, trying to reinitialise. '
                    f"Fail count: {fail_count}",
                )
                await self.release_resources()
                await asyncio.sleep(1)
                await self.initialise_stream()

                # Exit after too many failures
                if fail_count >= 10:
                    print('Too many failures. Exiting.')
                    break
                continue
            else:
                # Reset fail count on successful read
                fail_count = 0

                # Mark as successfully captured
                self.successfully_captured = True

            # Process the frame if the capture interval has elapsed
            current_time = datetime.datetime.now()
            elapsed_time = (current_time - last_process_time).total_seconds()

            # If the capture interval has elapsed, yield the frame
            if elapsed_time >= self.capture_interval:
                last_process_time = current_time
                timestamp = current_time.timestamp()
                yield frame, timestamp

                # Clear memory
                del frame, timestamp
                gc.collect()

            await asyncio.sleep(0.01)  # Adjust the sleep time as needed

        await self.release_resources()


    def update_capture_interval(self, new_interval: int) -> None:
        """
        Updates the capture interval.

        Args:
            new_interval (int): Frame capture interval in seconds.
        """
        self.capture_interval = new_interval


async def main():
    parser = argparse.ArgumentParser(
        description='Capture video frames asynchronously from webcam or stream.',
    )
    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help='Video source: webcam index (0, 1, etc.)',
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=15,
        help='Capture interval in seconds (default: 15)',
    )
    args = parser.parse_args()

    # Convert source to int if it's a webcam index
    source = args.source
    if source.isdigit():
        source = int(source)

    stream_capture = StreamCapture(source, args.interval)
    print(f"Starting capture from source: {source}")
    print(f"Capture interval: {args.interval} seconds")
    print("Press Ctrl+C to stop")

    try:
        async for frame, timestamp in stream_capture.execute_capture():
            # Process the frame here
            dt = datetime.datetime.fromtimestamp(timestamp)
            print(f"Frame captured at {dt.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Frame shape: {frame.shape}")

            # Optional: Display the frame (uncomment if you want to see it)
            # cv2.imshow('Captured Frame', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            # Release the frame resources
            del frame
            gc.collect()
    except KeyboardInterrupt:
        print("\nStopping capture...")
        await stream_capture.release_resources()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    asyncio.run(main())