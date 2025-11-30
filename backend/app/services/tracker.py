from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


def _centroid(rect: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = rect
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])


@dataclass
class CentroidTracker:
    max_disappeared: int = 15
    next_object_id: int = 0
    objects: Dict[int, np.ndarray] = field(default_factory=dict)
    disappeared: Dict[int, int] = field(default_factory=dict)

    def register(self, rect: np.ndarray) -> None:
        self.objects[self.next_object_id] = rect
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id: int) -> None:
        self.objects.pop(object_id, None)
        self.disappeared.pop(object_id, None)

    def update(self, rects: List[np.ndarray]) -> Dict[int, np.ndarray]:
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return dict(self.objects)

        if len(self.objects) == 0:
            for rect in rects:
                self.register(rect)
            return dict(self.objects)

        object_ids = list(self.objects.keys())
        object_centroids = np.array([_centroid(self.objects[object_id]) for object_id in object_ids])
        input_centroids = np.array([_centroid(rect) for rect in rects])

        distances = np.linalg.norm(object_centroids[:, np.newaxis] - input_centroids[np.newaxis, :], axis=2)

        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            object_id = object_ids[row]
            self.objects[object_id] = rects[col]
            self.disappeared[object_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(len(object_ids))) - used_rows
        unused_cols = set(range(len(rects))) - used_cols

        if len(object_ids) >= len(rects):
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
        else:
            for col in unused_cols:
                self.register(rects[col])

        return dict(self.objects)

