"""Classical pupil tracker — a fast screener over many videos.

Purpose is triage, not measurement of record. It runs in well under a minute per
video on CPU, where SAM 2 needs tens of minutes, so it can rank every candidate
video by drop depth; SAM 2 then confirms only the shortlist.

Segmentation uses the **red reflex**, not darkness. Under coaxial microscope
illumination the pupil is the brightest, reddest disc in the eye; the darkest
pixels inside the limbus are the limbus rim and the instruments, which is why a
darkest-percentile threshold returns a ring (measured circularity 0.22) rather
than the pupil.

The threshold is **Otsu on the a\\* (red-green) channel inside the limbus**, not a
fixed percentile. A relative percentile would pin the measured area to a constant
fraction of the limbus and suppress the very changes this detector exists to
find.

Per frame, inside a limbus circle fixed from the first frame:

1. Otsu-threshold the a\\* channel to separate the red pupil from the iris;
2. keep the largest connected component;
3. fit an ellipse and record its area;
4. record circularity, ellipse fill ratio and specular fraction, so the same
   occlusion filter and persistence rule used with SAM 2 apply unchanged.

Output columns match `scripts/12_track_pupil_sam2.py`, so
`pupil_measured.read_area_csv` reads either without special-casing.

Pure OpenCV/NumPy, no network (CLAUDE.md rule 2).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

SPECULAR_LEVEL = 230
MIN_CIRCULARITY = 0.55      # a clean mask on a 360x270 frame measures ~0.70
MIN_FILL_RATIO = 0.80       # blob area over fitted-ellipse area; an instrument bites this down
MAX_SPECULAR_FRACTION = 0.12
MIN_AREA_PX = 30
# A maximally dilated pupil is about 8 mm across against a ~12 mm limbus, so the
# area ratio cannot plausibly exceed ~0.45. Allowing 0.60 leaves room for Hough
# error in the limbus radius while still rejecting frames where the threshold has
# escaped the pupil and taken in the iris or the whole red field.
MAX_PUPIL_LIMBUS_RATIO = 0.60


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


def find_limbus(frame: np.ndarray) -> Limbus:
    """Limbus circle from the first frame, by Hough with a centred prior."""
    g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (9, 9), 2)
    h, w = g.shape
    circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(h, w) // 4,
                               param1=100, param2=30,
                               minRadius=int(min(h, w) * 0.08), maxRadius=int(min(h, w) * 0.45))
    if circles is not None:
        c = sorted(np.round(circles[0]).astype(int),
                   key=lambda c: abs(c[0] - w // 2) + abs(c[1] - h // 2))
        cx, cy, r = c[0]
        return Limbus(int(cx), int(cy), int(r))
    return Limbus(w // 2, h // 2, int(min(h, w) * 0.35))


def measure_frame(frame: np.ndarray, limbus: Limbus) -> dict:
    """Pupil ellipse area, circularity, fill ratio and specular fraction for one frame."""
    inside = limbus.mask(frame.shape).astype(bool)
    if not inside.any():
        return _empty("limbus mask empty")

    a = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 1]
    vals = a[inside]
    if vals.size == 0 or vals.max() == vals.min():
        return _empty("no red-green contrast")

    # Otsu over the pixels inside the limbus only, then applied to the full plane.
    cut, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    red = ((a >= cut) & inside).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(red, 8)
    if n <= 1:
        return _empty("no red component")
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    blob_area = int(stats[best, cv2.CC_STAT_AREA])
    if blob_area < MIN_AREA_PX:
        return _empty("component too small")
    blob = (labels == best).astype(np.uint8)

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
        "reject_reason": "",
    }


def judge(row: dict, limbus_px: int | None = None) -> str:
    """Occlusion and plausibility rules, matching those applied to the SAM 2 trace."""
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
    if row["specular_fraction"] > MAX_SPECULAR_FRACTION:
        return f"specular {row['specular_fraction']:.2f}"
    return ""


def _empty(reason: str) -> dict:
    return {
        "pupil_px": 0, "circularity": 0.0, "fill_ratio": 0.0,
        "specular_fraction": 0.0, "instrument_overlap": 0.0, "reject_reason": reason,
    }
