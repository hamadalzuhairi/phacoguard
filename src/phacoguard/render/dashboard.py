"""Composite the PhacoGuard dashboard over a surgical video frame.

Layout (1920x1080): the video sits on the left, the panels on the right, and a
persistent footer states that the indicators come from dataset labels.

Wording rules (CLAUDE.md rule 4): panels report observations and a risk state.
No string in this module tells the surgeon to do anything, and the word
"guidance" does not appear in any rendered text.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from phacoguard.data.phase_map import PHASES, SURGICAL_STEPS, TRANSITION
from phacoguard.mock.sources import MarkerStatus
from phacoguard.mock.timeline import (
    MEASURED_BANNER, MOCK_BANNER, NO_PHACO_LABELS, NO_PHASE_LABELS,
    NO_PUPIL_ANNOTATION, NO_RADIAL_SOURCE, CaseTimeline,
)

W, H = 1920, 1080
VIDEO_W = 1200
PANEL_X = VIDEO_W + 24
PANEL_W = W - PANEL_X - 24
FOOTER_H = 78

BG = (18, 18, 20)
PANEL = (30, 31, 36)
LINE = (56, 58, 66)
TEXT = (232, 233, 238)
MUTED = (150, 153, 163)
DIM = (96, 99, 110)

STATE_COLOUR = {
    "routine": (110, 178, 120),
    "rising": (70, 170, 235),
    "high": (78, 92, 232),
}
# Colour carries one meaning only: OBSERVED marks that something was observed, and the
# red of STATE_COLOUR["high"] is reserved for the case risk state. A lit marker beside a
# "routine" band would otherwise read as a contradiction.
OBSERVED = (70, 170, 235)
PHASE_COLOUR = {
    "incision": (150, 120, 90),
    "ovd_injection": (120, 105, 75),
    "capsulorhexis": (170, 140, 70),
    "hydrodissection": (140, 160, 80),
    "phaco": (80, 150, 200),
    "cortex_removal": (120, 130, 190),
    "capsule_polishing": (155, 120, 205),
    "iol_insertion": (150, 110, 160),
    "ovd_removal": (110, 95, 140),
    "wound_closure": (95, 120, 120),
    TRANSITION: (62, 64, 72),
}
STATUS_COLOUR = {
    "completed": (150, 153, 163),
    "in progress": (70, 170, 235),
    "upcoming": (96, 99, 110),
    "not performed": (74, 78, 88),
}
MARKER_LABEL = {
    "pupil_constriction": "Pupil constriction",
    "radial_folds": "Anterior capsule radial folds",
    "prolonged_phaco": "Prolonged phaco time",
}

F = cv2.FONT_HERSHEY_SIMPLEX
FD = cv2.FONT_HERSHEY_DUPLEX


def render_frame(timeline: CaseTimeline, frame: np.ndarray | None, t_s: float) -> np.ndarray:
    """One composited dashboard frame for case time `t_s`."""
    canvas = np.full((H, W, 3), BG, np.uint8)
    _video(canvas, timeline, frame, t_s)
    y = 24
    y = _risk_band(canvas, timeline, t_s, y)
    y = _step_list(canvas, timeline, t_s, y)
    y = _phase_bar(canvas, timeline, t_s, y)
    y = _phaco_meter(canvas, timeline, t_s, y)
    y = _indicators(canvas, timeline, t_s, y)
    _event_log(canvas, timeline, t_s, y)
    _footer(canvas, timeline)
    return canvas


def _video(canvas: np.ndarray, tl: CaseTimeline, frame: np.ndarray | None, t_s: float) -> None:
    area_h = H - FOOTER_H - 118
    box = (24, 118, VIDEO_W - 48, area_h - 24)
    cv2.rectangle(canvas, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), PANEL, -1)

    if frame is not None:
        fh, fw = frame.shape[:2]
        scale = min(box[2] / fw, box[3] / fh)
        new = cv2.resize(frame, (max(1, int(fw * scale)), max(1, int(fh * scale))))
        ox = box[0] + (box[2] - new.shape[1]) // 2
        oy = box[1] + (box[3] - new.shape[0]) // 2
        canvas[oy : oy + new.shape[0], ox : ox + new.shape[1]] = new
    else:
        _text(canvas, "no video frame available", (box[0] + 28, box[1] + box[3] // 2), 0.7, DIM)

    _text(canvas, "PhacoGuard", (24, 44), 1.0, TEXT, font=FD)
    _text(canvas, "research prototype - not a medical device", (250, 44), 0.55, MUTED)
    _text(
        canvas,
        f"{tl.dataset}  |  {tl.case_id}  |  {tl.licence}",
        (24, 78),
        0.62,
        MUTED,
    )
    _text(canvas, _clock(t_s), (VIDEO_W - 190, 78), 0.75, TEXT, font=FD)
    if tl.caption:
        _fit_text(canvas, tl.caption, (24, 104), VIDEO_W - 60, 0.46, (150, 190, 225))


def _risk_band(canvas: np.ndarray, tl: CaseTimeline, t_s: float, y: int) -> int:
    state = tl.state_at(t_s)
    colour = STATE_COLOUR[state]
    h = 88
    _panel(canvas, y, h)
    cv2.rectangle(canvas, (PANEL_X, y), (PANEL_X + 8, y + h), colour, -1)
    _text(canvas, "CASE RISK STATE", (PANEL_X + 26, y + 26), 0.46, MUTED)
    _text(canvas, state.upper(), (PANEL_X + 26, y + 70), 1.0, colour, font=FD)
    return y + h + 14


def _step_list(canvas: np.ndarray, tl: CaseTimeline, t_s: float, y: int) -> int:
    """The ten surgical steps in fixed order, with status and time spent.

    `transition` is shown as the current-phase label only. It is not a step, so
    it never takes a row: a case that is between steps has not performed an
    eleventh one.
    """
    rows = 21
    h = 38 + rows * len(SURGICAL_STEPS)
    _panel(canvas, y, h)
    reason = tl.unavailable.get("phase")
    _text(canvas, "SURGICAL STEPS", (PANEL_X + 26, y + 26), 0.48, MUTED)
    if reason:
        _reason(canvas, reason, y, h)
        return y + h + 14

    current = tl.phase_at(t_s)
    label = current.replace("_", " ")
    _text(canvas, f"now: {label}", (PANEL_X + 210, y + 26), 0.46,
          STATUS_COLOUR["in progress"] if current != TRANSITION else MUTED)

    top = y + 46
    for i, row in enumerate(tl.step_status(t_s)):
        ry = top + i * rows
        colour = STATUS_COLOUR[row["status"]]
        cv2.rectangle(canvas, (PANEL_X + 26, ry - 8), (PANEL_X + 30, ry + 4),
                      PHASE_COLOUR.get(row["step"], DIM), -1)
        _text(canvas, f"{row['index']:2d}", (PANEL_X + 38, ry + 3), 0.4, MUTED)
        _text(canvas, row["step"].replace("_", " "), (PANEL_X + 62, ry + 3), 0.42, colour)
        _text(canvas, row["status"], (PANEL_X + 250, ry + 3), 0.38, colour)
        if row["status"] != "not performed":
            _text(canvas, _clock(row["seconds"]), (PANEL_X + 385, ry + 3), 0.38, colour)
    return y + h + 14


def _phase_bar(canvas: np.ndarray, tl: CaseTimeline, t_s: float, y: int) -> int:
    h = 86
    _panel(canvas, y, h)
    _text(canvas, "STEP TIMELINE", (PANEL_X + 26, y + 24), 0.45, MUTED)
    if tl.unavailable.get("phase"):
        _reason(canvas, tl.unavailable["phase"], y, h)
        return y + h + 14

    x0, x1 = PANEL_X + 26, PANEL_X + PANEL_W - 26
    bar_y, bar_h = y + 34, 16
    total = max(tl.duration_s, 1e-6)
    cv2.rectangle(canvas, (x0, bar_y), (x1, bar_y + bar_h), (42, 44, 50), -1)
    present = []
    for i in tl.intervals:
        a = x0 + int((x1 - x0) * i.start_s / total)
        b = x0 + int((x1 - x0) * i.end_s / total)
        cv2.rectangle(canvas, (a, bar_y), (max(b, a + 1), bar_y + bar_h),
                      PHASE_COLOUR.get(i.phase, DIM), -1)
        if i.phase not in present:
            present.append(i.phase)
    cur = x0 + int((x1 - x0) * min(t_s, total) / total)
    cv2.line(canvas, (cur, bar_y - 5), (cur, bar_y + bar_h + 5), TEXT, 2)

    # Legend: the steps this case actually contains, in surgical order.
    lx, ly = x0, bar_y + bar_h + 20
    for step in [p for p in SURGICAL_STEPS if p in present]:
        name = step.replace("_", " ")
        (tw, _), _ = cv2.getTextSize(name, F, 0.33, 1)
        if lx + 12 + tw > x1:
            lx, ly = x0, ly + 15
        cv2.rectangle(canvas, (lx, ly - 6), (lx + 7, ly + 1), PHASE_COLOUR.get(step, DIM), -1)
        _text(canvas, name, (lx + 11, ly + 1), 0.33, MUTED)
        lx += 11 + tw + 12
    return y + h + 14


def _phaco_meter(canvas: np.ndarray, tl: CaseTimeline, t_s: float, y: int) -> int:
    h = 112
    _panel(canvas, y, h)
    _text(canvas, "PHACO TIME OBSERVED", (PANEL_X + 26, y + 30), 0.5, MUTED)
    reason = tl.unavailable.get("phaco")
    if reason:
        _reason(canvas, reason, y, h)
        return y + h + 16
    elapsed = sum(
        max(0.0, min(t_s, i.end_s) - i.start_s) for i in tl.intervals if i.phase == "phaco"
    )
    p85, p95 = tl.distribution["p85"], tl.distribution["p95"]
    ceiling = max(p95 * 1.25, elapsed * 1.1, 1.0)

    colour = OBSERVED if elapsed >= p85 else TEXT
    _text(canvas, _clock(elapsed), (PANEL_X + 26, y + 66), 0.82, colour, font=FD)
    _text(
        canvas,
        f"p85 {p85:.0f}s   p95 {p95:.0f}s   n={tl.distribution['n_cases']} labelled cases",
        (PANEL_X + 150, y + 64),
        0.45,
        MUTED,
    )

    x0, x1 = PANEL_X + 26, PANEL_X + PANEL_W - 26
    bar_y, bar_h = y + 86, 18
    cv2.rectangle(canvas, (x0, bar_y), (x1, bar_y + bar_h), (42, 44, 50), -1)
    fill = x0 + int((x1 - x0) * min(elapsed / ceiling, 1.0))
    cv2.rectangle(canvas, (x0, bar_y), (max(fill, x0 + 1), bar_y + bar_h), colour, -1)
    for cut, lbl in ((p85, "p85"), (p95, "p95")):
        mx = x0 + int((x1 - x0) * min(cut / ceiling, 1.0))
        cv2.line(canvas, (mx, bar_y - 5), (mx, bar_y + bar_h + 5), MUTED, 1)
        _text(canvas, lbl, (mx - 12, bar_y + bar_h + 22), 0.4, MUTED)
    return y + h + 16


def _indicators(canvas: np.ndarray, tl: CaseTimeline, t_s: float, y: int) -> int:
    h = 34 + 48 * len(MARKER_LABEL)
    _panel(canvas, y, h)
    _text(canvas, "MARKERS", (PANEL_X + 26, y + 24), 0.45, MUTED)
    row = y + 40
    for marker, label in MARKER_LABEL.items():
        result = tl.markers.get(marker)
        fired = [e for e in tl.events if e.type == marker and e.t_s <= t_s]
        unavailable = result is not None and result.status is MarkerStatus.NO_LABELLED_SOURCE

        if unavailable:
            colour = DIM
            status = (result.note or "no labelled source")
        elif fired:
            colour = OBSERVED
            measured = result is not None and result.status is MarkerStatus.MEASURED
            kind = "measured" if measured else "observed"
            status = f"{kind} at {_clock(fired[-1].t_s)}"
            if measured and len(fired) > 1:
                status += f"  ({len(fired)} episodes)"
        else:
            colour, status = (74, 78, 88), "not observed"

        cv2.circle(canvas, (PANEL_X + 38, row + 12), 7, colour, -1)
        _text(canvas, label, (PANEL_X + 56, row + 10), 0.48, TEXT if not unavailable else MUTED)
        _fit_text(canvas, status, (PANEL_X + 56, row + 30), PANEL_W - 86, 0.4, MUTED)
        row += 48
    return y + h + 14


def _event_log(canvas: np.ndarray, tl: CaseTimeline, t_s: float, y: int) -> None:
    h = H - FOOTER_H - 20 - y
    if h < 70:
        return
    _panel(canvas, y, h)
    _text(canvas, "EVENT LOG  (case time)", (PANEL_X + 26, y + 24), 0.45, MUTED)

    entries = [(e.t_s, "marker", MARKER_LABEL.get(e.type, e.type), e.observation)
               for e in tl.events if e.t_s <= t_s]
    entries += [(t, "step", step_line(i, step), "")
                for i, (t, step) in enumerate(tl.phase_changes(), start=1) if t <= t_s]
    entries.sort(key=lambda e: e[0])

    row = y + 50
    if not entries:
        _text(canvas, "no events recorded", (PANEL_X + 26, row), 0.44, DIM)
        return
    for t, kind, title, detail in entries[-9:]:
        if row > y + h - 22:
            break
        colour = TEXT if kind == "marker" else MUTED
        _fit_text(canvas, f"{_clock(t)}  {title}", (PANEL_X + 26, row), PANEL_W - 52, 0.43, colour)
        row += 18
        if detail:
            for line in _wrap(detail, 58)[:2]:
                _fit_text(canvas, line, (PANEL_X + 40, row), PANEL_W - 66, 0.38, MUTED)
                row += 16
        row += 6


def step_line(n: int, step: str) -> str:
    """Event-log wording for a step change."""
    return f"Step {n}: {step.replace('_', ' ')} started"


def _footer(canvas: np.ndarray, tl: CaseTimeline) -> None:
    y = H - FOOTER_H
    cv2.rectangle(canvas, (0, y), (W, H), (26, 27, 31), -1)
    cv2.line(canvas, (0, y), (W, y), LINE, 1)
    _text(canvas, tl.banner or MOCK_BANNER, (24, y + 30), 0.56, (120, 190, 240), font=FD)
    source = tl.footer_source or (
        f"Source: {tl.dataset} {tl.case_id} ({tl.licence}). "
        f"Phase and phaco-time events derive from the dataset's expert phase labels."
    )
    _fit_text(canvas, source, (24, y + 58), W - 60, 0.46, MUTED)


def _reason(canvas: np.ndarray, reason: str, y: int, h: int) -> None:
    """State why a panel has no data. A blank panel would read as a negative finding."""
    lines = _wrap(reason, 46)[:3]
    top = y + 62 - 11 * (len(lines) - 1)
    for i, line in enumerate(lines):
        _text(canvas, line, (PANEL_X + 26, top + 22 * i), 0.47, DIM)


def _fit_text(canvas, s: str, org, max_w: int, scale: float, colour) -> None:
    """Draw text, shrinking until it fits the given width."""
    while scale > 0.28:
        (w, _), _ = cv2.getTextSize(s, F, scale, 1)
        if w <= max_w:
            break
        scale -= 0.02
    _text(canvas, s, org, scale, colour)


def _panel(canvas: np.ndarray, y: int, h: int) -> None:
    cv2.rectangle(canvas, (PANEL_X, y), (PANEL_X + PANEL_W, y + h), PANEL, -1)
    cv2.rectangle(canvas, (PANEL_X, y), (PANEL_X + PANEL_W, y + h), LINE, 1)


def _text(canvas, s, org, scale, colour, font=F, thickness=1) -> None:
    cv2.putText(canvas, s, org, font, scale, colour, thickness, cv2.LINE_AA)


def _wrap(s: str, width: int) -> list[str]:
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def _clock(t_s: float) -> str:
    t = max(0, int(round(t_s)))
    return f"{t // 60:02d}:{t % 60:02d}"


def assert_observational(strings: list[str]) -> None:
    """Guard for CLAUDE.md rule 4. Raises if rendered text instructs or says 'guidance'."""
    banned = (
        "guidance", "you should", "you must", "please ", "make sure",
        "consider ", "recommend", "advise", "perform ", "switch to", "increase ", "reduce ",
    )
    for s in strings:
        low = s.lower()
        for token in banned:
            if token in low:
                raise AssertionError(f"non-observational text: {s!r} contains {token!r}")


def ui_strings() -> list[str]:
    """Every fixed string this module renders, for the rule 4 test."""
    return [
        "PhacoGuard", "research prototype - not a medical device", "CASE RISK STATE",
        "SURGICAL PHASE", "PHACO TIME OBSERVED", "MARKERS", "EVENT LOG",
        "no events recorded", "no video frame available", "not observed",
        "SURGICAL STEPS", "STEP TIMELINE", "completed", "in progress", "upcoming",
        "not performed", *(step_line(i, p) for i, p in enumerate(SURGICAL_STEPS, 1)),
        "EVENT LOG  (case time)", NO_PHASE_LABELS, NO_PHACO_LABELS,
        NO_PUPIL_ANNOTATION, NO_RADIAL_SOURCE, MEASURED_BANNER,
        "no labelled source", MOCK_BANNER, *MARKER_LABEL.values(), *PHASES,
    ]
