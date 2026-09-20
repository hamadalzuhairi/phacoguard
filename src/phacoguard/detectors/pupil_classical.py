"""Classical pupil tracker — a fast screener over many videos.

Purpose is triage, not measurement of record. It runs in well under a minute per
video on CPU, where SAM 2 needs tens of minutes, so it can rank every candidate
video by drop depth; SAM 2 then confirms only the shortlist.

Two things vary during a cataract operation and both are tracked rather than
assumed.

**Segmentation mode.** Under coaxial illumination a pupil with a red reflex is
the brightest, reddest disc in the eye; a pupil without one is the darkest. In a
dense cataract the reflex is absent early and appears once the nucleus is gone,
so the mode is decided per window from the a\\* contrast between the central disc
and the surrounding annulus, and is allowed to switch mid-case. The switch is
recorded: it marks the point at which the nucleus stopped blocking the reflex.

**Limbus.** The eye moves and the microscope zooms, so the limbus circle is
re-estimated every few seconds, smoothed over recent accepted estimates, and a
new estimate that jumps too far from the previous one is discarded.

Pure OpenCV/NumPy, no network (CLAUDE.md rule 2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

MIN_CIRCULARITY = 0.55
MIN_FILL_RATIO = 0.80
MIN_AREA_PX = 30
# A maximally dilated pupil is about 8 mm across against a ~12 mm limbus, so the
# area ratio cannot plausibly exceed ~0.45. Allowing 0.60 leaves room for error in
# the limbus radius while still rejecting frames where the threshold has escaped
# the pupil and taken in the iris or the whole red field.
MAX_PUPIL_LIMBUS_RATIO = 0.60

# The specular test is deliberately absent. Over 30,103 screened frames it
# rejected 3, while circularity rejected 14,587: it was dead weight presenting
# itself as a safeguard. `specular_fraction` is still recorded for the record.
SPECULAR_LEVEL = 230

MODE_WINDOW_S = 20.0          # re-decide red vs dark this often
# Hysteresis. A single threshold at 4.0 sat in the middle of the contrast
# distribution and the mode flapped: case_709 switched 6+ times on contrasts of
# 1-19. RED is entered only above the high mark, DARK only below the low one,
# and in between the previous mode stands.
MODE_CONTRAST_HIGH = 6.0
MODE_CONTRAST_LOW = 2.0
# Otsu runs inside this fraction of the limbus radius. The full disc includes the
# white sclera ring and the limbal specular highlights, which dominate the
# histogram: in dark mode Otsu then split sclera from everything else and
# returned the whole iris plus pupil (measured area ratio 0.86-0.99).
SEGMENT_ROI_FRACTION = 0.80
LIMBUS_INTERVAL_S = 3.0       # re-estimate the limbus this often
LIMBUS_MAX_JUMP = 0.20        # reject a radius or centre jump beyond this fraction
LIMBUS_SMOOTH_N = 5           # median over this many accepted estimates


class Mode(str, Enum):
    RED = "red"
    DARK = "dark"


@dataclass(frozen=True)
class Limbus:
    cx: int
    cy: int
    r: int

    @property
    def area_px(self) -> int:
        return int(np.pi * self.r * self.r)

    def mask(self, shape: tuple[int, int]) -> np.ndarray:
        m = np.zeros(shape[:2], np.uint8)
        cv2.circle(m, (self.cx, self.cy), self.r, 255, -1)
        return m


@dataclass
class ModeSwitch:
    t_s: float
    from_mode: Mode
    to_mode: Mode
    contrast: float

    def as_dict(self) -> dict:
        return {"t_s": round(self.t_s, 2), "from": self.from_mode.value,
                "to": self.to_mode.value, "a_contrast": round(self.contrast, 2)}


def hough_limbus(frame: np.ndarray) -> Limbus | None:
    """A single limbus estimate from one frame, or None when Hough finds nothing."""
    g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (9, 9), 2)
    h, w = g.shape
    circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(h, w) // 4,
                               param1=100, param2=30,
                               minRadius=int(min(h, w) * 0.08), maxRadius=int(min(h, w) * 0.45))
    if circles is None:
        return None
    c = sorted(np.round(circles[0]).astype(int),
               key=lambda c: abs(c[0] - w // 2) + abs(c[1] - h // 2))
    cx, cy, r = c[0]
    return Limbus(int(cx), int(cy), int(r))


def default_limbus(shape: tuple[int, int]) -> Limbus:
    h, w = shape[:2]
    return Limbus(w // 2, h // 2, int(min(h, w) * 0.35))


@dataclass
class LimbusTracker:
    """Re-estimates the limbus periodically, smoothed, rejecting implausible jumps."""

    interval_s: float = LIMBUS_INTERVAL_S
    max_jump: float = LIMBUS_MAX_JUMP
    smooth_n: int = LIMBUS_SMOOTH_N
    current: Limbus | None = None
    _history: list[Limbus] = field(default_factory=list)
    _last_t: float = -1e9
    n_accepted: int = 0
    n_rejected: int = 0

    def update(self, frame: np.ndarray, t_s: float) -> Limbus:
        """Return the limbus to use at `t_s`, re-estimating when due."""
        if self.current is None:
            self.current = hough_limbus(frame) or default_limbus(frame.shape)
            self._history = [self.current]
            self._last_t = t_s
            self.n_accepted += 1
            return self.current
        if t_s - self._last_t < self.interval_s:
            return self.current
        self._last_t = t_s
        candidate = hough_limbus(frame)
        if candidate is None or not self._plausible(candidate):
            self.n_rejected += 1
            return self.current
        self.n_accepted += 1
        self._history.append(candidate)
        self._history = self._history[-self.smooth_n:]
        self.current = Limbus(
            int(np.median([l.cx for l in self._history])),
            int(np.median([l.cy for l in self._history])),
            int(np.median([l.r for l in self._history])),
        )
        return self.current

    def _plausible(self, candidate: Limbus) -> bool:
        prev = self.current
        assert prev is not None
        if abs(candidate.r - prev.r) / max(prev.r, 1) > self.max_jump:
            return False
        shift = float(np.hypot(candidate.cx - prev.cx, candidate.cy - prev.cy))
        return shift / max(prev.r, 1) <= self.max_jump


def a_contrast(frame: np.ndarray, limbus: Limbus) -> float:
    """Median a\\* of the central disc minus that of the surrounding annulus.

    Positive means the centre is redder than the iris around it, i.e. a red
    reflex is present.
    """
    a = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 1].astype(np.int16)
    h, w = a.shape
    centre = np.zeros((h, w), np.uint8)
    cv2.circle(centre, (limbus.cx, limbus.cy), max(int(limbus.r * 0.45), 2), 255, -1)
    outer = np.zeros((h, w), np.uint8)
    cv2.circle(outer, (limbus.cx, limbus.cy), int(limbus.r * 0.95), 255, -1)
    cv2.circle(outer, (limbus.cx, limbus.cy), int(limbus.r * 0.60), 0, -1)
    c_vals, o_vals = a[centre.astype(bool)], a[outer.astype(bool)]
    if c_vals.size == 0 or o_vals.size == 0:
        return 0.0
    return float(np.median(c_vals) - np.median(o_vals))


def decide_mode(frames: list[np.ndarray], limbus: Limbus,
                previous: "Mode | None" = None,
                high: float = MODE_CONTRAST_HIGH,
                low: float = MODE_CONTRAST_LOW) -> tuple[Mode, float]:
    """Red or dark for this window, with hysteresis around the previous mode."""
    if not frames:
        return previous or Mode.RED, 0.0
    median = float(np.median([a_contrast(f, limbus) for f in frames]))
    if median >= high:
        return Mode.RED, median
    if median <= low:
        return Mode.DARK, median
    return (previous or (Mode.RED if median >= (high + low) / 2 else Mode.DARK)), median


def segment_pupil(frame: np.ndarray, limbus: Limbus, mode: Mode) -> np.ndarray | None:
    """Largest candidate pupil component under the given mode."""
    roi = Limbus(limbus.cx, limbus.cy, max(int(limbus.r * SEGMENT_ROI_FRACTION), 3))
    inside = roi.mask(frame.shape).astype(bool)
    if not inside.any():
        return None
    if mode is Mode.RED:
        plane = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 1]
        take_above = True
    else:
        plane = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        take_above = False
    vals = plane[inside]
    if vals.size == 0 or vals.max() == vals.min():
        return None
    cut, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # OpenCV's Otsu returns t with foreground defined as src > t, so the
    # comparison must be strict. With >= the threshold could land on the iris
    # value and take the iris in with the pupil.
    sel = (plane > cut) if take_above else (plane <= cut)
    blob = (sel & inside).astype(np.uint8)
    blob = cv2.morphologyEx(blob, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(blob, 8)
    if n <= 1:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if int(stats[best, cv2.CC_STAT_AREA]) < MIN_AREA_PX:
        return None
    return (labels == best).astype(np.uint8)


def measure_frame(frame: np.ndarray, limbus: Limbus, mode: Mode = Mode.RED) -> dict:
    """Pupil ellipse area, circularity, fill ratio and specular fraction."""
    blob = segment_pupil(frame, limbus, mode)
    if blob is None:
        return _empty("no pupil component")
    blob_area = int(blob.sum())
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _empty("no contour")
    contour = max(contours, key=cv2.contourArea)

    if len(contour) >= 5:
        (_, _), (ax1, ax2), _ = cv2.fitEllipse(contour)
        ellipse_area = float(np.pi * ax1 * ax2 / 4.0)
    else:
        ellipse_area = float(blob_area)
    fill = blob_area / ellipse_area if ellipse_area > 0 else 0.0
    perimeter = cv2.arcLength(contour, True)
    circ = float(4 * np.pi * cv2.contourArea(contour) / (perimeter ** 2)) if perimeter else 0.0
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    specular = float((grey[blob.astype(bool)] > SPECULAR_LEVEL).sum()) / max(blob_area, 1)

    return {
        "pupil_px": int(round(ellipse_area)),
        "circularity": round(min(circ, 1.0), 4),
        "fill_ratio": round(min(fill, 1.0), 4),
        "specular_fraction": round(specular, 4),
        "instrument_overlap": 0.0,
        "mode": mode.value,
        "reject_reason": "",
    }


def judge(row: dict, limbus_px: int | None = None) -> str:
    """Occlusion and plausibility rules. No specular test — see the module docstring."""
    if row["pupil_px"] <= 0:
        return row.get("reject_reason") or "no pupil"
    if limbus_px:
        ratio = row["pupil_px"] / limbus_px
        if ratio > MAX_PUPIL_LIMBUS_RATIO:
            return f"implausible pupil/limbus {ratio:.2f}"
    if row["circularity"] < MIN_CIRCULARITY:
        return f"circularity {row['circularity']:.2f}"
    if row.get("fill_ratio", 1.0) < MIN_FILL_RATIO:
        return f"fill {row['fill_ratio']:.2f}"
    return ""


def _empty(reason: str) -> dict:
    return {
        "pupil_px": 0, "circularity": 0.0, "fill_ratio": 0.0, "specular_fraction": 0.0,
        "instrument_overlap": 0.0, "mode": "", "reject_reason": reason,
    }
