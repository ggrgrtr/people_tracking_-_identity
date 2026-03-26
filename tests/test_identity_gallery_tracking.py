import unittest
from unittest import mock

import numpy as np

from identity_gallery_tracking.app import ThreadedCameraCapture
from identity_gallery_tracking.config import AppConfig
from identity_gallery_tracking.identity_manager import IdentityManager
from identity_gallery_tracking.tracklets import Tracklet, TrackletTracker


class _DummyCapture:
    def read(self):
        return False, None


class _TrackStub:
    def __init__(self, observed=True, person_id=None):
        self.person_id = person_id
        self.identity_feature = np.array([1.0, 0.0], dtype=np.float32)
        self.identity_color_hist = np.array([1.0, 0.0], dtype=np.float32)
        self.identity_face_feature = None
        self.smooth_bbox = (10, 10, 60, 120)
        self._observed = observed

    def is_confirmed(self, min_hits=None):
        return True

    def identity_ready(self):
        return True

    def was_observed(self):
        return self._observed


class IdentityGalleryTrackingTests(unittest.TestCase):
    def test_threaded_capture_falls_back_when_threading_backend_is_unavailable(self):
        capture = ThreadedCameraCapture(_DummyCapture())

        with mock.patch("identity_gallery_tracking.app.threading._start_new_thread", new=None, create=True):
            started = capture.start()

        self.assertIsNone(started)
        self.assertIsNone(capture.thread)

    def test_threaded_capture_distinguishes_timeout_and_eof(self):
        capture = ThreadedCameraCapture(_DummyCapture())

        status, frame, timestamp = capture.read(timeout=0.01)
        self.assertEqual(status, "timeout")
        self.assertIsNone(frame)
        self.assertIsNone(timestamp)

        capture.read_failed = True
        status, frame, timestamp = capture.read(timeout=0.01)
        self.assertEqual(status, "eof")
        self.assertIsNone(frame)
        self.assertIsNone(timestamp)

    def test_identity_manager_skips_unobserved_tracklets(self):
        config = AppConfig()
        manager = IdentityManager(config)
        frame_shape = (240, 320, 3)

        observed_track = _TrackStub(observed=True)
        manager.observe_tracklets(
            [observed_track],
            frame_id=1,
            elapsed_seconds=0.25,
            frame_shape=frame_shape,
            update_gallery=True,
        )

        identity = manager.identities[observed_track.person_id]
        self.assertEqual(identity.observations, 1)
        self.assertEqual(identity.last_seen_frame, 1)

        predicted_track = _TrackStub(observed=False, person_id=observed_track.person_id)
        manager.observe_tracklets(
            [predicted_track],
            frame_id=2,
            elapsed_seconds=0.50,
            frame_shape=frame_shape,
            update_gallery=False,
        )

        self.assertEqual(identity.observations, 1)
        self.assertEqual(identity.last_seen_frame, 1)

    def test_deduplicate_returns_removed_tracklets(self):
        config = AppConfig()
        tracker = TrackletTracker(config)
        frame_shape = (240, 320, 3)
        feature = np.array([1.0, 0.0], dtype=np.float32)
        color_hist = np.array([1.0, 0.0], dtype=np.float32)

        track_one = Tracklet(1, (10, 10, 60, 120), feature, color_hist, None, frame_shape, config)
        track_two = Tracklet(2, (12, 12, 60, 120), feature.copy(), color_hist.copy(), None, frame_shape, config)
        tracker.active_tracklets = [track_one, track_two]

        removed = tracker.predict_only(frame_shape)

        self.assertEqual(len(removed), 1)
        self.assertEqual(len(tracker.active_tracklets), 1)
        self.assertEqual(tracker.active_tracklets[0].id, 1)


if __name__ == "__main__":
    unittest.main()
