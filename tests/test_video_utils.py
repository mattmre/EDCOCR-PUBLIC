"""Tests for shared video sampling helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import video_utils as mod


class TestBuildVideoFramePlan:
    def test_builds_deterministic_time_sampled_plan(self):
        plan = mod.build_video_frame_plan(
            frame_count=300,
            fps=30.0,
            sample_seconds=1.0,
            max_frames=20,
        )
        assert plan == [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 299]

    def test_caps_large_plan_to_max_frames(self):
        plan = mod.build_video_frame_plan(
            frame_count=10_000,
            fps=30.0,
            sample_seconds=0.25,
            max_frames=25,
        )
        assert len(plan) <= 25
        assert plan[0] == 0
        assert plan[-1] == 9999

    def test_invalid_fps_falls_back_to_one(self):
        plan = mod.build_video_frame_plan(
            frame_count=5,
            fps=0.0,
            sample_seconds=1.0,
            max_frames=10,
        )
        assert plan == [0, 1, 2, 3, 4]


class _FakeCapture:
    def __init__(self, frames: list[np.ndarray], fps: float = 10.0):
        self.frames = frames
        self.fps = fps
        self.position = 0
        self.released = False

    def isOpened(self):
        return True

    def release(self):
        self.released = True

    def get(self, prop_id):
        if prop_id == 1:
            return float(self.position)
        if prop_id == 5:
            return float(self.fps)
        if prop_id == 7:
            return float(len(self.frames))
        return 0.0

    def set(self, prop_id, value):
        if prop_id == 1:
            self.position = int(value)
            return True
        return False

    def read(self):
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame.copy()


class TestVideoFrameIteration:
    def test_get_video_page_count_uses_sampling_plan(self, monkeypatch):
        capture = _FakeCapture(
            [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(95)],
            fps=10.0,
        )
        monkeypatch.setattr(
            mod,
            "cv2",
            SimpleNamespace(
                CAP_PROP_POS_FRAMES=1,
                CAP_PROP_FPS=5,
                CAP_PROP_FRAME_COUNT=7,
                VideoCapture=lambda _path: capture,
            ),
        )
        monkeypatch.setattr(mod, "_CV2_AVAILABLE", True)

        assert mod.get_video_page_count("clip.mp4", sample_seconds=1.0, max_frames=20) == 11
        assert capture.released is True

    def test_iter_video_frames_returns_requested_sampled_frames(self, monkeypatch):
        frames = []
        for idx in range(10):
            frame = np.zeros((4, 4, 3), dtype=np.uint8)
            frame[:, :, 0] = idx
            frame[:, :, 1] = idx + 1
            frame[:, :, 2] = idx + 2
            frames.append(frame)

        captures = []

        def _factory(_path):
            capture = _FakeCapture(frames, fps=2.0)
            captures.append(capture)
            return capture

        monkeypatch.setattr(
            mod,
            "cv2",
            SimpleNamespace(
                CAP_PROP_POS_FRAMES=1,
                CAP_PROP_FPS=5,
                CAP_PROP_FRAME_COUNT=7,
                VideoCapture=_factory,
            ),
        )
        monkeypatch.setattr(mod, "_CV2_AVAILABLE", True)

        images = list(mod.iter_video_frames("clip.mp4", start=2, end=4, sample_seconds=1.0, max_frames=10))

        assert len(images) == 3
        assert images[0].mode == "RGB"
        assert images[0].getpixel((0, 0)) == (4, 3, 2)
        assert images[1].getpixel((0, 0)) == (6, 5, 4)
        assert images[2].getpixel((0, 0)) == (8, 7, 6)
        assert captures[-1].released is True

    def test_iter_video_frames_raises_when_opencv_missing(self, monkeypatch):
        monkeypatch.setattr(mod, "_CV2_AVAILABLE", False)
        monkeypatch.setattr(mod, "cv2", None)

        with pytest.raises(RuntimeError, match="OpenCV video support not available"):
            list(mod.iter_video_frames("clip.mp4", start=1, end=1))
