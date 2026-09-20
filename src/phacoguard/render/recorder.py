"""Write the dashboard over a case video to an mp4, and chain cases into one demo.

Frames are composited deterministically rather than screen-captured, so the
recording is reproducible and the same layout is reused when model outputs
replace the dataset labels.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from phacoguard.mock.timeline import CaseTimeline
from phacoguard.render.dashboard import H, W, render_frame


class DemoWriter:
    """An mp4 writer that several cases can be appended to in turn."""

    def __init__(self, path: Path, fps: float = 25.0):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.fps = fps
        self._writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        if not self._writer.isOpened():
            raise RuntimeError(f"could not open {path} for writing")
        self.frames_written = 0

    def add_case(self, timeline: CaseTimeline, speed: float = 1.0, max_seconds: float | None = None) -> int:
        """Composite one case. `speed` > 1 plays the case faster to keep the demo short."""
        cap = cv2.VideoCapture(str(timeline.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"could not open video {timeline.video_path}")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or timeline.fps
        limit = max_seconds if max_seconds is not None else timeline.duration_s
        written = 0
        try:
            t = 0.0
            step = speed / self.fps
            while t < limit:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ok, frame = cap.read()
                if not ok:
                    break
                self._writer.write(render_frame(timeline, frame, t))
                written += 1
                t += step
        finally:
            cap.release()
        self.frames_written += written
        return written

    def add_card(self, lines: list[str], seconds: float = 2.5) -> int:
        """A plain title card between cases, naming the next dataset and case."""
        canvas = np.full((H, W, 3), (18, 18, 20), np.uint8)
        y = H // 2 - 22 * len(lines)
        for i, line in enumerate(lines):
            scale = 1.0 if i == 0 else 0.6
            colour = (232, 233, 238) if i == 0 else (150, 153, 163)
            (tw, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_DUPLEX, scale, 1)
            cv2.putText(canvas, line, ((W - tw) // 2, y), cv2.FONT_HERSHEY_DUPLEX, scale,
                        colour, 1, cv2.LINE_AA)
            y += 52 if i == 0 else 34
        n = int(seconds * self.fps)
        for _ in range(n):
            self._writer.write(canvas)
        self.frames_written += n
        return n

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> "DemoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
