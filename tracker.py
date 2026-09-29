"""
tracker.py - Centroid Tracker with Multi-Frame Confirmation to eliminate flicker & false alarms.
"""
import math
from typing import List, Tuple, Dict


class PotholeTracker:
    def __init__(self, min_confirmed_frames: int = 3, max_disappeared: int = 15, distance_threshold: float = 80.0):
        self.next_object_id = 1
        self.objects: Dict[int, Tuple[int, int]] = {}
        self.disappeared: Dict[int, int] = {}
        self.consecutive_counts: Dict[int, int] = {}
        self.logged_objects = set()
        self.min_confirmed_frames = min_confirmed_frames
        self.max_disappeared = max_disappeared
        self.distance_threshold = distance_threshold

    def register(self, centroid: Tuple[int, int]) -> int:
        obj_id = self.next_object_id
        self.objects[obj_id] = centroid
        self.disappeared[obj_id] = 0
        self.consecutive_counts[obj_id] = 1
        self.next_object_id += 1
        return obj_id

    def deregister(self, obj_id: int):
        self.objects.pop(obj_id, None)
        self.disappeared.pop(obj_id, None)
        self.consecutive_counts.pop(obj_id, None)

    def update(self, rects: List[Tuple[int, int, int, int]]) -> List[Tuple[int, Tuple[int, int, int, int], bool]]:
        if len(rects) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return []

        input_centroids = [((x1 + x2) // 2, (y1 + y2) // 2) for (x1, y1, x2, y2) in rects]

        if len(self.objects) == 0:
            results = []
            for i, rect in enumerate(rects):
                obj_id = self.register(input_centroids[i])
                ready = (self.consecutive_counts[obj_id] >= self.min_confirmed_frames)
                results.append((obj_id, rect, ready))
            return results

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        distances = []
        for i, (ox, oy) in enumerate(object_centroids):
            for j, (ix, iy) in enumerate(input_centroids):
                d = math.hypot(ox - ix, oy - iy)
                distances.append((d, i, j))
        distances.sort(key=lambda x: x[0])

        used_rows, used_cols, matched = set(), set(), []
        for d, row, col in distances:
            if row in used_rows or col in used_cols or d > self.distance_threshold:
                continue
            obj_id = object_ids[row]
            self.objects[obj_id] = input_centroids[col]
            self.disappeared[obj_id] = 0
            self.consecutive_counts[obj_id] += 1
            used_rows.add(row)
            used_cols.add(col)
            
            ready_to_log = (self.consecutive_counts[obj_id] >= self.min_confirmed_frames)
            matched.append((obj_id, rects[col], ready_to_log))

        for j in range(len(input_centroids)):
            if j not in used_cols:
                obj_id = self.register(input_centroids[j])
                ready_to_log = (self.consecutive_counts[obj_id] >= self.min_confirmed_frames)
                matched.append((obj_id, rects[j], ready_to_log))

        for i in range(len(object_ids)):
            if i not in used_rows:
                obj_id = object_ids[i]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)

        return matched
