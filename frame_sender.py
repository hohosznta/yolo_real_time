from __future__ import annotations

import logging
import os

import numpy as np
from dotenv import load_dotenv

from src.utils import Utils

# Load environment variables from .env file
load_dotenv()


class BackendFrameSender:
    """
    A class to send frames and metadata to a backend server via HTTP or
    WebSocket.

    Attributes:
        base_url (str): The base URL of the backend API.
        shared_token (dict[str, str | bool]):
            Shared token state for authentication.
        max_retries (int):
            Maximum number of retries for HTTP requests.
        timeout (int):
            Timeout in seconds for HTTP requests.
    """

    def __init__(
        self,
        api_url: str | None = None,
        max_retries: int = 3,
        timeout: int = 10,
        shared_token: dict[str, str | bool] | None = None,
    ) -> None:
        """
        Initialise the BackendFrameSender.

        Args:
            max_retries: Maximum number of retries for HTTP requests.
            timeout: Timeout in seconds for HTTP requests.
            shared_token: Shared token state for authentication.
        """
        # Set the base URL from the provided api_url or environment variable
        if api_url is None:
            api_url = os.getenv('KAKAO_API_URL', 'https://kapi.kakao.com/v2/api/talk/memo/default/send')
        # Remove trailing slash for consistency
        self.base_url: str = api_url.rstrip('/')

        if shared_token is not None:
            self.shared_token = shared_token
        else:
            self.shared_token = {
                'access_token': '',
            }

        # Basic settings
        self.max_retries: int = max_retries
        self.timeout: int = timeout

        # Suppress httpx debug logging for cleaner output
        logging.getLogger('httpx').setLevel(logging.WARNING)

    async def send_optimized_frame(
        self: BackendFrameSender,
        frame: np.ndarray,
        site: str,
        stream_name: str,
        encoding_format: str = 'jpeg',
        jpeg_quality: int = 85,
        warnings_json: str = '',
        cone_polygons_json: str = '',
        pole_polygons_json: str = '',
        detection_items_json: str = '',
    ) -> dict:
        """
        Send frame with optimized encoding to backend service.

        Args:
            frame (np.ndarray): Frame to send
            site (str): Site identifier
            stream_name (str): Stream identifier
            encoding_format (str): Image encoding format ('jpeg' or 'png')
            jpeg_quality (int): JPEG quality (1-100) when using JPEG format
            warnings_json (str): JSON string of warnings
            cone_polygons_json (str): JSON string of cone polygons
            pole_polygons_json (str): JSON string of pole polygons
            detection_items_json (str): JSON string of detection items

        Returns:
            dict: Response from backend
        """
        try:
            # Encode frame based on specified format for optimal compression
            if encoding_format.lower() == 'jpeg':
                encoded_frame = Utils.encode_frame(frame, 'jpeg', jpeg_quality)
            else:
                encoded_frame = Utils.encode_frame(frame, 'png')

            if not encoded_frame:
                logging.error('Failed to encode frame')
                return {'success': False, 'error': 'Failed to encode frame'}

            height, width = frame.shape[:2]

            return await self.send_frame(
                    site=site,
                    stream_name=stream_name,
                    frame_bytes=encoded_frame,
                    warnings_json=warnings_json,
                    cone_polygons_json=cone_polygons_json,
                    pole_polygons_json=pole_polygons_json,
                    detection_items_json=detection_items_json,
                    width=width,
                    height=height,
                )

        except Exception as e:
            logging.error('Error sending optimized frame: %s', e)
            return {'success': False, 'error': str(e)}

    async def send_frame(
        self: BackendFrameSender,
        site: str,
        stream_name: str,
        frame_bytes: bytes,
        warnings_json: str = '',
        cone_polygons_json: str = '',
        pole_polygons_json: str = '',
        detection_items_json: str = '',
        width: int = 0,
        height: int = 0,
    ) -> dict:
        """
        Send a frame and metadata to the backend via HTTP POST.

        Args:
            site (str): The site/label name.
            stream_name (str): The stream key or camera identifier.
            frame_bytes (bytes): The raw image bytes to upload.
            warnings_json (str, optional):
                JSON string of warnings. Defaults to ''.
            cone_polygons_json (str, optional):
                JSON string of cone polygons. Defaults to ''.
            pole_polygons_json (str, optional):
                JSON string of pole polygons. Defaults to ''.
            detection_items_json (str, optional):
                JSON string of detection items. Defaults to ''.
            width (int, optional): Frame width. Defaults to 0.
            height (int, optional): Frame height. Defaults to 0.

        Returns:
            dict: The JSON response from the backend.

        Raises:
            RuntimeError: If all retry attempts are exhausted.
            httpx.HTTPStatusError: For non-401 HTTP errors.
            Exception: For other unexpected errors.
    """
        # Prepare multipart/form-data for the image and metadata
        files: dict[str, tuple[str, bytes, str]] = {
            'file': (
                'frame.jpg',  # Use .jpg extension for consistency
                frame_bytes,
                'image/jpeg',  # JPEG MIME type for better compression
            ),
        }
        # Prepare the POST data payload
        data: dict[str, str | int] = {
            'label': site,
            'key': stream_name,
            'warnings_json': warnings_json,
            'cone_polygons_json': cone_polygons_json,
            'pole_polygons_json': pole_polygons_json,
            'detection_items_json': detection_items_json,
            'width': width,
            'height': height,
        }

        # Use shared NetClient for HTTP with retries
        return await self._net.http_post(
            '/frames',
            object_type="text",
            text = files,
            max_retries=self.max_retries,
        )

    async def close(self: BackendFrameSender) -> None:
        """
        Close connection prevent resource leaks
        and thread issues.
        """
        try:
            await self._net.close()
        except Exception as e:
            logging.error('Error closing NetClient: %s', e)