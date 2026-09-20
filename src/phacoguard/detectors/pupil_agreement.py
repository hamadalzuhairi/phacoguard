"""Agreement between the classical screener and the SAM 2 confirmation run.

The screener ranks; SAM 2 confirms. This module says whether the two agree, both
on the area series and on the events found, so a shortlist drawn by the screener
is not trusted on its own.

Pure arithmetic, no network (CLAUDE.md rule 2).
"""
from __future__ import annotations

from dataclasses import dataclass

from phacoguard.detectors.pupil_measured import AreaSample, DropEvent, drop_events

DEFAULT_MATCH_TOLERANCE_S = 15.0


@dataclass(frozen=True)
class Agreement:
    n_paired: int
    pearson_r: float | None
    classical_events: list[DropEvent]
    sam2_events: list[DropEvent]
    matched: list[tuple[float, float, float]]   # (classical t_s, sam2 t_s, |delta|)
    classical_only: list[float]
    sam2_only: list[float]

    @property
    def verdict(self) -> str:
        if self.n_paired < 10 or self.pearson_r is None:
            return "insufficient overlap to judge"
        if self.pearson_r >= 0.8 and self.matched and not self.sam2_only:
            return "good agreement"
        if self.pearson_r >= 0.6 and self.matched:
            return "partial agreement"
        return "poor agreement"

    def as_dict(self) -> dict:
        return {
            "n_paired_samples": self.n_paired,
            "pearson_r": round(self.pearson_r, 4) if self.pearson_r is not None else None,
            "n_classical_events": len(self.classical_events),
            "n_sam2_events": len(self.sam2_events),
            "matched_events": [
                {"classical_t_s": round(a, 1), "sam2_t_s": round(b, 1), "delta_s": round(d, 1)}
                for a, b, d in self.matched
            ],
            "classical_only_t_s": [round(t, 1) for t in self.classical_only],
            "sam2_only_t_s": [round(t, 1) for t in self.sam2_only],
            "verdict": self.verdict,
        }


def compare(classical: list[AreaSample], sam2: list[AreaSample],
            tolerance_s: float = DEFAULT_MATCH_TOLERANCE_S, **detector_kw) -> Agreement:
    """Correlate the two area series and match their events."""
    paired = _pair_on_time(classical, sam2, tolerance_s=1.0)
    r = _pearson([a for a, _ in paired], [b for _, b in paired])

    c_events = drop_events(classical, **detector_kw)
    s_events = drop_events(sam2, **detector_kw)

    matched, used = [], set()
    for ce in c_events:
        best, best_d = None, None
        for k, se in enumerate(s_events):
            if k in used:
                continue
            d = abs(se.t_s - ce.t_s)
            if d <= tolerance_s and (best_d is None or d < best_d):
                best, best_d = k, d
        if best is not None:
            used.add(best)
            matched.append((ce.t_s, s_events[best].t_s, best_d))

    matched_c = {a for a, _, _ in matched}
    return Agreement(
        n_paired=len(paired),
        pearson_r=r,
        classical_events=c_events,
        sam2_events=s_events,
        matched=matched,
        classical_only=[e.t_s for e in c_events if e.t_s not in matched_c],
        sam2_only=[e.t_s for k, e in enumerate(s_events) if k not in used],
    )


def _pair_on_time(a: list[AreaSample], b: list[AreaSample],
                  tolerance_s: float) -> list[tuple[float, float]]:
    """Nearest-neighbour pairing on timestamp, clean samples only, one use each."""
    bs = sorted((s for s in b if s.clean), key=lambda s: s.t_s)
    out, j = [], 0
    for s in sorted((s for s in a if s.clean), key=lambda s: s.t_s):
        while j + 1 < len(bs) and abs(bs[j + 1].t_s - s.t_s) < abs(bs[j].t_s - s.t_s):
            j += 1
        if j < len(bs) and abs(bs[j].t_s - s.t_s) <= tolerance_s:
            out.append((s.normalised_area, bs[j].normalised_area))
    return out


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / (sxx ** 0.5 * syy ** 0.5)
