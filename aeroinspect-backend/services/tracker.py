"""
tracker.py
----------
Lightweight centroid-based object tracker.
Assigns consistent IDs to detections across frames using IoU matching.
Used as a fallback when ByteTrack / SORT are unavailable.
"""

from collections import OrderedDict
import numpy as np


def _iou(boxA, boxB):
    """Compute IoU between two [x, y, w, h] boxes."""
    ax1, ay1 = boxA[0], boxA[1]
    ax2, ay2 = boxA[0] + boxA[2], boxA[1] + boxA[3]
    bx1, by1 = boxB[0], boxB[1]
    bx2, by2 = boxB[0] + boxB[2], boxB[1] + boxB[3]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    areaA = (ax2 - ax1) * (ay2 - ay1)
    areaB = (bx2 - bx1) * (by2 - by1)
    union = areaA + areaB - inter_area

    return inter_area / union if union > 0 else 0.0


class CentroidTracker:
    """
    Assigns persistent IDs to detections using IoU overlap matching.
    Detections that overlap sufficiently with a tracked object keep its ID.
    New detections that don't match any existing object get a new ID.
    Objects that disappear for more than max_disappeared frames are removed.
    """

    def __init__(self, max_disappeared=10, iou_threshold=0.3):
        self.next_id       = 1
        self.objects       = OrderedDict()   # id → bbox [x,y,w,h]
        self.disappeared   = OrderedDict()   # id → frames since last seen
        self.max_disappeared = max_disappeared
        self.iou_threshold   = iou_threshold

    def _register(self, bbox):
        self.objects[self.next_id]     = bbox
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def _deregister(self, oid):
        del self.objects[oid]
        del self.disappeared[oid]

    def update(self, detections):
        """
        detections: list of [x, y, w, h] boxes for the current frame.
        Returns:    dict { track_id: bbox }
        """
        # No detections — age all existing objects
        if len(detections) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)
            return dict(self.objects)

        # No existing tracks — register all detections
        if len(self.objects) == 0:
            for bbox in detections:
                self._register(bbox)
            return dict(self.objects)

        # Match detections to existing tracks via IoU
        object_ids   = list(self.objects.keys())
        object_bboxes = list(self.objects.values())

        iou_matrix = np.zeros((len(object_bboxes), len(detections)))
        for r, ob in enumerate(object_bboxes):
            for c, db in enumerate(detections):
                iou_matrix[r, c] = _iou(ob, db)

        # Greedy matching: best IoU first
        matched_tracks = set()
        matched_dets   = set()

        flat_order = np.argsort(-iou_matrix, axis=None)
        for idx in flat_order:
            r, c = divmod(idx, len(detections))
            if iou_matrix[r, c] < self.iou_threshold:
                break
            if r in matched_tracks or c in matched_dets:
                continue
            oid = object_ids[r]
            self.objects[oid]     = detections[c]
            self.disappeared[oid] = 0
            matched_tracks.add(r)
            matched_dets.add(c)

        # Age unmatched tracks
        for r, oid in enumerate(object_ids):
            if r not in matched_tracks:
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)

        # Register new detections
        for c in range(len(detections)):
            if c not in matched_dets:
                self._register(detections[c])

        return dict(self.objects)

    def reset(self):
        self.next_id = 1
        self.objects.clear()
        self.disappeared.clear()
