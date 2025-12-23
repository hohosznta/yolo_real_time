from __future__ import annotations

import asyncio
import base64
import logging
import math
import os
from datetime import datetime

import cv2
import networkx as nx
import numpy as np
from shapely.geometry import LineString
from shapely.geometry import MultiPoint
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.ops import unary_union
from sklearn.cluster import HDBSCAN
from watchdog.events import FileSystemEventHandler


class Utils:
    """
    A class to provide utility functions.
    """

    # ---- Geometry/tuning constants for utility pole area construction ----
    CIRCLE_BUFFER_SEGMENTS: int = 64
    TANGENT_BUFFER_WIDTH: float = 0.05
    TANGENT_BUFFER_SEGMENTS: int = 32
    UTILITY_POLE_RADIUS_FACTOR: float = 0.35

    @staticmethod
    def is_expired(expire_date_str: str | None) -> bool:
        """
        Check if the given expire date string is expired.

        Args:
            expire_date_str (str | None): The expire date string
                in ISO 8601 format.

        Returns:
            bool: True if expired, False otherwise.
        """
        if expire_date_str:
            try:
                expire_date = datetime.fromisoformat(expire_date_str)
                return datetime.now() > expire_date
            except ValueError:
                # If the string cannot be parsed as a valid ISO 8601 date
                return False
        return False

    @staticmethod
    def encode(value: str) -> str:
        """
        Encode a value into a URL-safe Base64 string.

        Args:
            value (str): The value to encode.

        Returns:
            str: The encoded string.
        """
        return base64.urlsafe_b64encode(
            value.encode('utf-8'),
        ).decode('utf-8')

    @staticmethod
    def encode_frame(
        frame: np.ndarray, format: str = 'jpeg', quality: int = 85,
    ) -> bytes:
        """
        Encodes an image frame (NumPy array) into specified format as bytes
        with compression.

        Args:
            frame (np.ndarray): The image frame to encode. Should be a valid
                NumPy array representing an image, typically in BGR format
                as used by OpenCV.
            format (str): Image format ('jpeg' or 'png'). JPEG is recommended
                for video streams.
            quality (int): For JPEG: quality level (0-100), for PNG:
                compression level (0-9).

        Returns:
            bytes: The encoded image as bytes. Returns an empty bytes
                object if encoding fails.

        Raises:
            None: All exceptions are caught and logged; function returns
                b'' on error.
        """
        try:
            if format.lower() == 'jpeg':
                # JPEG compression - much smaller file size for video streams
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                success, buffer = cv2.imencode('.jpg', frame, encode_params)
            else:
                # PNG compression - lossless but larger
                encode_params = [
                    cv2.IMWRITE_PNG_COMPRESSION, min(quality // 10, 9),
                ]
                success, buffer = cv2.imencode('.png', frame, encode_params)

            if not success:
                logging.error(
                    f"Failed to encode frame - OpenCV imencode returned False "
                    f"for {format}",
                )
                return b''
            return buffer.tobytes()
        except Exception as e:
            logging.error(f"Error encoding frame as {format}: {e}")
            return b''

    @staticmethod
    def create_h264_encoder(
        width: int,
        height: int,
        fps: int = 30,
        bitrate: int = 2000000,
    ) -> cv2.VideoWriter | None:
        """
        Create an H.264 video encoder for streaming.

        Args:
            width (int): Frame width
            height (int): Frame height
            fps (int): Frames per second
            bitrate (int): Target bitrate in bits per second

        Returns:
            cv2.VideoWriter | None: Video writer instance or None if failed
        """
        try:
            # H.264 codec with hardware acceleration if available
            fourcc = cv2.VideoWriter_fourcc(*'H264')

            # Create video writer for H.264 stream
            gst_pipeline = (
                'appsrc ! videoconvert ! x264enc bitrate={} '
                'speed-preset=ultrafast tune=zerolatency ! h264parse ! '
                'rtph264pay config-interval=1 pt=96 ! '
                'udpsink host=127.0.0.1 port=5000'
            ).format(bitrate // 1000)

            writer = cv2.VideoWriter(
                gst_pipeline,
                fourcc,
                fps,
                (width, height),
            )

            if writer.isOpened():
                return writer
            else:
                logging.error('Failed to initialize H.264 encoder')
                return None

        except Exception as e:
            logging.error(f"Error creating H.264 encoder: {e}")
            return None

    @staticmethod
    def encode_frame_h264(frame: np.ndarray, encoder: cv2.VideoWriter) -> bool:
        """
        Encode and write a frame using H.264 encoder.

        Args:
            frame (np.ndarray): Frame to encode
            encoder (cv2.VideoWriter): H.264 encoder instance

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if encoder and encoder.isOpened():
                encoder.write(frame)
                return True
            return False
        except Exception as e:
            logging.error(f"Error encoding H.264 frame: {e}")
            return False

    @staticmethod
    def filter_warnings_by_working_hour(
        warnings: dict[str, dict[str, int]],
        is_working_hour: bool,
    ) -> dict[str, dict[str, int]]:
        """
        Filters the warnings dictionary according to working hours.

        Args:
            warnings (dict[str, dict[str, int]]):
                Dictionary of warning types and their parameters, e.g.:
                {
                    "warning_people_in_controlled_area": {"count": 3},
                    "warning_no_safety_vest": {},
                    ...
                }
            is_working_hour (bool):
                Whether the current time is within working hours.

        Returns:
            dict[str, dict[str, int]]:
                Filtered warnings dictionary, containing only relevant warnings
                according to the working hour status.

        Notes:
            - During working hours, all warnings are returned.
            - Outside working hours, only 'warning_people_in_controlled_area'
              is retained.
            - If no warnings are present, returns an empty dictionary.
        """
        # If no warnings exist, return an empty dictionary immediately.
        if not warnings:
            return {}

        # During working hours, return all warnings without filtering.
        if is_working_hour:
            return warnings

        # Outside working hours, retain only warnings related to controlled
        # areas.
        filtered: dict[str, dict[str, int]] = {}
        for key, params in warnings.items():
            # Only keep 'warning_people_in_controlled_area' for
            # notification/storage.
            if key == 'warning_people_in_controlled_area':
                filtered[key] = params

        return filtered

    @staticmethod
    def should_notify(
        timestamp: int,
        last_notification_time: int,
        cooldown_period: int = 300,
    ) -> bool:
        """
        Check if a notification should be sent based on the cooldown period.

        Args:
            timestamp (int): The current timestamp.
            last_notification_time (int):
                The timestamp of the last notification.
            cooldown_period (int): The cooldown period in seconds.

        Returns:
            bool: True if a notification should be sent, False otherwise.
        """
        return (timestamp - last_notification_time) >= cooldown_period

    @staticmethod
    def normalise_bbox(bbox: list[float]) -> list[float]:
        """
        Normalises the bounding box coordinates.

        Args:
            bbox (List[float]): The bounding box coordinates.

        Returns:
            List[float]: Normalised coordinates.
        """
        left_x = min(bbox[0], bbox[2])
        right_x = max(bbox[0], bbox[2])
        top_y = min(bbox[1], bbox[3])
        bottom_y = max(bbox[1], bbox[3])
        if len(bbox) > 4:
            return [left_x, top_y, right_x, bottom_y, bbox[4], bbox[5]]
        return [left_x, top_y, right_x, bottom_y]

    @staticmethod
    def normalise_data(datas: list[list[float]]) -> list[list[float]]:
        """
        Normalises a list of bounding box data.

        Args:
            datas (List[List[float]]): List of bounding box data.

        Returns:
            List[List[float]]: Normalised data.
        """
        return [Utils.normalise_bbox(data[:4] + data[4:]) for data in datas]

    @staticmethod
    def overlap_percentage(bbox1: list[float], bbox2: list[float]) -> float:
        """
        Calculate the overlap percentage between two bounding boxes.

        Args:
            bbox1 (List[float]): The first bounding box.
            bbox2 (List[float]): The second bounding box.

        Returns:
            float: The overlap percentage.
        """
        # Calculate the coordinates of the intersection rectangle
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        # Calculate the area of the intersection rectangle
        overlap_area = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate the area of both bounding boxes
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        # Calculate the overlap percentage
        return overlap_area / float(area1 + area2 - overlap_area)

    @staticmethod
    def is_driver(person_bbox: list[float], vehicle_bbox: list[float]) -> bool:
        """
        Check if a person is a driver based on position near a vehicle.

        Args:
            person_bbox (List[float]): Bounding box of person.
            vehicle_bbox (List[float]): Bounding box of vehicle.

        Returns:
            bool: True if the person is likely the driver, False otherwise.
        """
        # Extract coordinates and dimensions of person and vehicle boxes
        person_bottom_y = person_bbox[3]
        person_top_y = person_bbox[1]
        person_left_x = person_bbox[0]
        person_right_x = person_bbox[2]
        person_width = person_bbox[2] - person_bbox[0]
        person_height = person_bbox[3] - person_bbox[1]

        vehicle_top_y = vehicle_bbox[1]
        vehicle_bottom_y = vehicle_bbox[3]
        vehicle_left_x = vehicle_bbox[0]
        vehicle_right_x = vehicle_bbox[2]
        vehicle_height = vehicle_bbox[3] - vehicle_bbox[1]

        # 1. Check vertical bottom position: person's bottom should be above
        #    the vehicle's bottom by at least half the person's height
        if not (
            person_bottom_y < vehicle_bottom_y
            and vehicle_bottom_y - person_bottom_y >= person_height / 2
        ):
            return False

        # 2. Check horizontal position: person's edges should not extend
        #    beyond half the width of the person from the vehicle's edges
        if not (
            person_left_x >= vehicle_left_x - person_width / 2
            and person_right_x <= vehicle_right_x + person_width / 2
        ):
            return False

        # 3. The person's top must be below the vehicle's top
        if not (person_top_y > vehicle_top_y):
            return False

        # 4. Person's height is less than or equal to half the vehicle's height
        if not (person_height <= vehicle_height / 2):
            return False

        return True

    @staticmethod
    def is_dangerously_close(
        person_bbox: list[float],
        vehicle_bbox: list[float],
        label: str,
    ) -> bool:
        """
        Determine if a person is dangerously close to machinery or vehicles.

        Args:
            person_bbox (list[float]): Bounding box of person.
            vehicle_bbox (list[float]): Machine/vehicle box.
            label (str): Type of the second object ('machinery' or 'vehicle').

        Returns:
            bool: True if the person is dangerously close, False otherwise.
        """
        # Calculate dimensions of the person bounding box
        person_width = person_bbox[2] - person_bbox[0]
        person_height = person_bbox[3] - person_bbox[1]
        person_area = person_width * person_height

        # Calculate the area of the vehicle bounding box
        vehicle_area = (vehicle_bbox[2] - vehicle_bbox[0]) * \
            (vehicle_bbox[3] - vehicle_bbox[1])
        acceptable_ratio = 0.1 if label == 'vehicle' else 0.05

        # Check if person area ratio is acceptable compared to vehicle area
        if person_area / vehicle_area > acceptable_ratio:
            return False

        # Define danger distances
        danger_distance_horizontal = 5 * person_width
        danger_distance_vertical = 1.5 * person_height

        # Calculate min horizontal/vertical distance between person and
        # vehicle
        horizontal_distance = min(
            abs(person_bbox[2] - vehicle_bbox[0]),
            abs(person_bbox[0] - vehicle_bbox[2]),
        )
        vertical_distance = min(
            abs(person_bbox[3] - vehicle_bbox[1]),
            abs(person_bbox[1] - vehicle_bbox[3]),
        )

        # Determine if the person is dangerously close
        return (
            horizontal_distance <= danger_distance_horizontal and
            vertical_distance <= danger_distance_vertical
        )

    @staticmethod
    def detect_polygon_from_cones(
        datas: list[list[float]],
        clusterer: HDBSCAN,
    ) -> list[Polygon]:
        """
        Detects polygons from the safety cones in the detection data.

        Args:
            datas (list[list[float]]): The detection data.

        Returns:
            list[Polygon]: A list of polygons formed by the safety cones.
        """
        if not datas:
            return []

        # Get positions of safety cones
        cone_positions = np.array([
            (
                (float(data[0]) + float(data[2])) / 2,
                (float(data[1]) + float(data[3])) / 2,
            )
            for data in datas if data[5] == 6
        ])

        # Check if there are at least three safety cones to form a polygon
        if len(cone_positions) < 3:
            return []

        # Cluster the safety cones
        labels = clusterer.fit_predict(cone_positions)

        # Extract clusters
        clusters: dict[int, list[np.ndarray]] = {}
        for point, label in zip(cone_positions, labels):
            if label == -1:
                continue  # Skip noise points
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(point)

        # Create polygons from clusters
        polygons = []
        for cluster_points in clusters.values():
            if len(cluster_points) >= 3:
                polygon = MultiPoint(cluster_points).convex_hull
                polygons.append(polygon)

        return polygons

    @staticmethod
    def calculate_people_in_controlled_area(
        polygons: list[Polygon],
        datas: list[list[float]],
    ) -> int:
        """
        Calculates the number of people within the safety cone area.

        Args:
            polygons (list[Polygon]): Polygons representing controlled areas.
            datas (list[list[float]]): The detection data.

        Returns:
            int: The number of people within the controlled area.
        """
        # Check if there are any detections
        if not datas:
            return 0

        # Check if there are valid polygons
        if not polygons:
            return 0

        # Use a set to track unique people
        unique_people = set()

        # Count the number of people within the controlled area
        for data in datas:
            if data[5] == 5:  # Check if it's a person
                x_center = (data[0] + data[2]) / 2
                y_center = (data[1] + data[3]) / 2
                point = Point(x_center, y_center)
                for polygon in polygons:
                    if polygon.contains(point):
                        # Update the set of unique people
                        unique_people.add((x_center, y_center))
                        break  # No need to check other polygons

        return len(unique_people)

    @staticmethod
    def build_utility_pole_union(
        datas: list[list[float]],
        clusterer: HDBSCAN,
    ) -> Polygon:
        """
        Builds a union Polygon representing the controlled area for utility
        poles.

        This method clusters detected utility poles, constructs minimum
        spanning trees (MST), calculates outer tangents, and unions the
        resulting polygons to form the final controlled area.

        Args:
            datas (list[list[float]]): Detection data, each entry is a list
                of floats representing bounding box and class info.
            clusterer (HDBSCAN): Clustering algorithm instance for grouping
                utility poles.

        Returns:
            Polygon: The union of all utility pole controlled areas. May be
                empty if no poles are detected.

        Notes:
            - If only one utility pole is detected, returns a single buffered
              circle.
            - If the number of poles is less than clusterer.min_samples,
              unions all circles directly.
            - Otherwise, clusters poles, builds MSTs, and unions circles and
              tangents.
        """
        # 1) Collect poles (centre-bottom and radius derived from bbox)
        utility_poles = Utils._extract_utility_poles(datas)
        if not utility_poles:
            return Polygon()

        # 2) Handle trivial cases
        if len(utility_poles) == 1:
            cx, cy, r = utility_poles[0]
            return Point(cx, cy).buffer(r, quad_segs=64)

        # 3) Too few poles for clustering → union all circles directly
        if len(utility_poles) < clusterer.min_samples:
            return Utils._union_circles(utility_poles)

        # 4) Cluster poles and build per-cluster unions
        clusters = Utils._cluster_utility_poles(utility_poles, clusterer)
        cluster_polys: list[Polygon] = [
            Utils._build_cluster_union(circles_in_cluster)
            for circles_in_cluster in clusters.values()
        ]
        return unary_union(cluster_polys)

    @staticmethod
    def _extract_utility_poles(
        datas: list[list[float]],
    ) -> list[tuple[float, float, float]]:
        """
        Extract utility pole centre-bottom and radius from detections.

        Args:
            datas (list[list[float]]): Detection data, each entry is a list
                of floats representing bounding box and class info.

        Returns:
            list[tuple[float, float, float]]: List of utility pole
                centres and radii (cx, cy, r).
        """
        poles: list[tuple[float, float, float]] = []
        for d in datas:
            if d[5] == 9:  # class == 9 => utility pole
                left, top, right, bottom, *_ = d
                cx: float = (left + right) / 2.0
                cy: float = bottom
                height: float = bottom - top
                radius: float = Utils.UTILITY_POLE_RADIUS_FACTOR * height
                if radius > 0:
                    poles.append((cx, cy, radius))
        return poles

    @staticmethod
    def _union_circles(poles: list[tuple[float, float, float]]) -> Polygon:
        """
        Union buffered circles for all given poles.

        Args:
            poles (list[tuple[float, float, float]]): List of utility pole
                centres and radii (cx, cy, r).

        Returns:
            Polygon: The union polygon for the cluster.
        """
        circle_polys: list[Polygon] = [
            Point(cx, cy).buffer(r, quad_segs=Utils.CIRCLE_BUFFER_SEGMENTS)
            for (cx, cy, r) in poles
        ]
        return unary_union(circle_polys)

    @staticmethod
    def _cluster_utility_poles(
        poles: list[tuple[float, float, float]],
        clusterer: HDBSCAN,
    ) -> dict[str | int, list[tuple[float, float, float]]]:
        """
        Cluster poles using HDBSCAN, grouping noise separately.

        Args:
            poles (list[tuple[float, float, float]]): List of utility pole
                centres and radii (cx, cy, r).
            clusterer (HDBSCAN): HDBSCAN clustering algorithm instance.

        Returns:
            dict[str | int, list[tuple[float, float, float]]]: Clusters of
                utility poles, keyed by cluster label.
        """
        # Prepare data for clustering
        coords: np.ndarray = np.array([(p[0], p[1]) for p in poles])
        labels: np.ndarray = clusterer.fit_predict(coords)

        # Map labels to original poles
        clusters: dict[str | int, list[tuple[float, float, float]]] = {}
        for idx, (circle, label) in enumerate(zip(poles, labels)):
            if label == -1:
                key: str = f"noise_{idx}"
                clusters.setdefault(key, []).append(circle)
            else:
                clusters.setdefault(int(label), []).append(circle)
        return clusters

    @staticmethod
    def _build_cluster_union(
        circles_in_cluster: list[tuple[float, float, float]],
    ) -> Polygon:
        """
        Build union polygon for a cluster of poles (circles + tangents).

        Args:
            circles_in_cluster (list[tuple[float, float, float]]):
                List of circles represented by (cx, cy, radius) tuples.

        Returns:
            Polygon: The union polygon for the cluster.
        """
        # Single pole: just a buffered circle
        if len(circles_in_cluster) == 1:
            cx, cy, r = circles_in_cluster[0]
            return Point(cx, cy).buffer(
                r, quad_segs=Utils.CIRCLE_BUFFER_SEGMENTS,
            )

        # Multiple poles: circles + outer tangents along MST edges
        circle_polys_: list[Polygon] = [
            Point(cx, cy).buffer(r, quad_segs=Utils.CIRCLE_BUFFER_SEGMENTS)
            for (cx, cy, r) in circles_in_cluster
        ]
        tangent_buffers = Utils._build_mst_tangent_buffers(circles_in_cluster)
        return unary_union(circle_polys_ + tangent_buffers)

    @staticmethod
    def _build_mst_tangent_buffers(
        circles_in_cluster: list[tuple[float, float, float]],
    ) -> list[Polygon]:
        """
        Build buffered polygons from outer tangents along MST edges.

        Args:
            circles_in_cluster (list[tuple[float, float, float]]):
                List of circles represented by (cx, cy, radius) tuples.

        Returns:
            list[Polygon]: List of buffered polygons.
        """
        mst_edges: list[tuple[int, int]] = Utils.build_mst_pairs(
            circles_in_cluster,
        )
        lines: list[LineString] = []
        for (u, v) in mst_edges:
            cx1, cy1, r1 = circles_in_cluster[u]
            cx2, cy2, r2 = circles_in_cluster[v]
            lines.extend(Utils.get_outer_tangents(cx1, cy1, r1, cx2, cy2, r2))
        return [
            ls.buffer(
                Utils.TANGENT_BUFFER_WIDTH,
                quad_segs=Utils.TANGENT_BUFFER_SEGMENTS,
            ) for ls in lines
        ]

    @staticmethod
    def build_mst_pairs(
        poles: list[tuple[float, float, float]],
    ) -> list[tuple[int, int]]:
        """
        Builds a minimum spanning tree (MST) for a set of utility poles.

        Args:
            poles (list[tuple[float, float, float]]): List of utility pole
                centres and radii (cx, cy, r).

        Returns:
            list[tuple[int, int]]: List of MST edges as index pairs.

        Notes:
            - Uses Euclidean distance minus radii as edge weights.
            - Returns edges as index pairs for use in tangent calculation.
        """
        G: nx.Graph = nx.Graph()
        for i, (cx, cy, r) in enumerate(poles):
            G.add_node(i, pos=(cx, cy), radius=r)

        n: int = len(poles)
        for i in range(n):
            cx1, cy1, r1 = poles[i]
            for j in range(i + 1, n):
                cx2, cy2, r2 = poles[j]
                dist_centers: float = math.dist((cx1, cy1), (cx2, cy2))
                weight: float = max(0, dist_centers - (r1 + r2))
                G.add_edge(i, j, weight=weight)

        mst: nx.Graph = nx.minimum_spanning_tree(G, weight='weight')
        return list(mst.edges())

    @staticmethod
    def get_outer_tangents(
        cx1: float,
        cy1: float,
        r1: float,
        cx2: float,
        cy2: float,
        r2: float,
        eps: float = 1e-9,
    ) -> list[LineString]:
        """
        Calculates the outer tangents between two circles.

        Args:
            cx1 (float): Centre x-coordinate of the first circle.
            cy1 (float): Centre y-coordinate of the first circle.
            r1 (float): Radius of the first circle.
            cx2 (float): Centre x-coordinate of the second circle.
            cy2 (float): Centre y-coordinate of the second circle.
            r2 (float): Radius of the second circle.
            eps (float): Small epsilon to avoid division by zero.

        Returns:
            list[LineString]: List of LineString objects representing outer
                tangents.

        Notes:
            - Returns empty list if circles overlap or are coincident.
            - Ensures r1 >= r2 for calculation stability.
        """
        dx: float = cx2 - cx1
        dy: float = cy2 - cy1
        d2: float = dx * dx + dy * dy
        d: float = math.sqrt(d2)
        if d < abs(r1 - r2):
            return []  # Circles overlap, no outer tangents
        if d < eps:
            return []  # Circles are coincident

        # Ensure r1 >= r2 for calculation stability
        if r2 > r1:
            cx1, cx2 = cx2, cx1
            cy1, cy2 = cy2, cy1
            r1, r2 = r2, r1
            dx, dy = -dx, -dy

        d2 = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2
        d = math.sqrt(d2)
        rdiff: float = r1 - r2
        if d < rdiff:
            return []  # Circles overlap

        alpha: float = math.acos(rdiff / d)
        theta: float = math.atan2((cy2 - cy1), (cx2 - cx1))

        lines: list[LineString] = []
        for sign in [1, -1]:
            phi: float = theta + sign * alpha
            x1t: float = cx1 + r1 * math.cos(phi)
            y1t: float = cy1 + r1 * math.sin(phi)
            x2t: float = cx2 + r2 * math.cos(phi)
            y2t: float = cy2 + r2 * math.sin(phi)
            ls: LineString = LineString([
                (x1t, y1t),
                (x2t, y2t),
            ])
            lines.append(ls)

        return lines

    @staticmethod
    def count_people_in_polygon(
        poly: Polygon,
        datas: list[list[float]],
    ) -> int:
        """
        Counts the number of people within a specified polygon.

        Args:
            poly (Polygon): The polygon representing the area of interest.
            datas (list[list[float]]): Detection data, each entry is a list
                of floats representing bounding box and class info.

        Returns:
            int: The number of unique people found within the polygon.

        Notes:
            - Only considers entries with class == 5 (person).
            - Uses centre point of bounding box for inclusion test.
        """
        persons: list[list[float]] = [d for d in datas if d[5] == 5]
        found_people: set[tuple[float, float]] = set()
        for p in persons:
            left, top, right, bottom, *_ = p
            px: float = (left + right) / 2.0
            py: float = (top + bottom) / 2.0
            if poly.contains(Point(px, py)):
                found_people.add((px, py))
        return len(found_people)

    @staticmethod
    def polygons_to_coords(polygons: list[Polygon]) -> list[list[list[float]]]:
        """
        Converts Polygon or MultiPolygon objects to a list of lists of
        [x, y] coordinates.

        Args:
            polygons (list[Polygon]): List of Polygon or MultiPolygon objects.

        Returns:
            list[list[list[float]]]: List of coordinate lists for each
                polygon.

        Notes:
            - Skips empty polygons.
            - For MultiPolygon, extracts coordinates from each sub-polygon.
        """
        coords_list: list[list[list[float]]] = []
        for poly in polygons:
            if poly.is_empty:
                continue  # Skip empty polygons
            if poly.geom_type == 'Polygon':
                coords_list.append([
                    list(pt) for pt in poly.exterior.coords
                ])
            elif poly.geom_type == 'MultiPolygon':
                for subpoly in poly.geoms:
                    if (
                        not subpoly.is_empty and
                        subpoly.geom_type == 'Polygon'
                    ):
                        coords_list.append([
                            list(pt)
                            for pt in subpoly.exterior.coords
                        ])
        return coords_list


class FileEventHandler(FileSystemEventHandler):
    """
    A class to handle file events.
    """

    def __init__(self, file_path: str, callback, loop):
        """
        Initialises the FileEventHandler instance.

        Args:
            file_path (str): The path of the file to watch.
            callback (Callable): The function to call when file is modified.
            loop (asyncio.AbstractEventLoop): The asyncio event loop.
        """
        self.file_path = os.path.abspath(file_path)
        self.callback = callback
        self.loop = loop

    def on_modified(self, event):
        """
        Called when a file is modified.

        Args:
            event (FileSystemEvent): The event object.
        """
        event_path = os.path.abspath(event.src_path)
        if event_path == self.file_path:
            print(f"[DEBUG] Configuration file modified: {event_path}")
            asyncio.run_coroutine_threadsafe(
                # Ensure the callback is run in the loop
                self.callback(), self.loop,
            )

