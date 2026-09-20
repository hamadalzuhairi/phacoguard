"""Measured pupil-area decrease.

This is a **measurement**, not an expert annotation. The pupil area comes from
SAM 2 zero-shot tracking (`scripts/12_track_pupil_sam2.py`); the 20% drop, the
30 s window and the 10 s persistence rule are the team's choices, not dataset
labels. Everything downstream describes it as "measured pupil-area decrease,
ungated (no phase labels for this case)" and never as clinician-labelled.

Occluded frames are excluded before any drop is computed: an instrument lying
across the pupil shrinks the visible mask and would otherwise read as a
constriction. A drop must also persist over clean frames for at least 10 s, so a
single bad reading cannot fire the indicator.

No network access; pure arithmetic over a time series (rule 2).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from phacoguard.data.phase_map import PUPIL_GATE_STEPS

DEFAULT_DROP = 0.20
DEFAULT_WINDOW_S = 30.0
DEFAULT_PERSIST_S = 10.0
# The occlusion filter removes roughly half of all frames, and the survivors arrive
# in clumps. A running maximum taken over a sparse, unevenly spaced window inflates
# the apparent drop, so a candidate is only counted when its window is well covered
# and is flanked by clean samples on both sides.
DEFAULT_MIN_WINDOW_COVERAGE = 0.60
DEFAULT_MIN_FLANK_SAMPLES = 5


@dataclass(frozen=True)
class AreaSample:
    t_s: float
    normalised_area: float
    clean: bool = True
    reject_reason: str = ""


@dataclass(frozen=True)
class DropEvent:
    """One sustained fall below the running maximum."""

    t_s: float
    drop_fraction: float
    max_drop_fraction: float
    window_max: float
    area: float
    sustained_s: float
    phase: str | None
    used_clean_frames: bool

    @property
    def observation(self) -> str:
        gate = "ungated" if self.phase is None else f"during {self.phase}"
        return (
            f"measured pupil area {self.max_drop_fraction * 100:.0f}% below its "
            f"{DEFAULT_WINDOW_S:.0f} s running maximum, sustained {self.sustained_s:.0f} s ({gate})"
        )


def read_area_csv(path: Path) -> list[AreaSample]:
    """Read the tracker's CSV, carrying the per-frame clean/occluded judgement."""
    out: list[AreaSample] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            area = float(row["normalised_pupil_area"])
            if area <= 0:
                continue
            out.append(AreaSample(
                t_s=float(row["t_s"]),
                normalised_area=area,
                clean=bool(int(row.get("clean", 1))),
                reject_reason=row.get("reject_reason", ""),
            ))
    out.sort(key=lambda s: s.t_s)
    return out


def drop_events(
    samples: list[AreaSample],
    drop: float = DEFAULT_DROP,
    window_s: float = DEFAULT_WINDOW_S,
    persist_s: float = DEFAULT_PERSIST_S,
    min_window_coverage: float = DEFAULT_MIN_WINDOW_COVERAGE,
    min_flank_samples: int = DEFAULT_MIN_FLANK_SAMPLES,
    use_clean_only: bool = True,
    phase_at=None,
    gate_phases: tuple[str, ...] = PUPIL_GATE_STEPS,
    rate_limit_s: float = 30.0,
) -> list[DropEvent]:
    """Flag each *sustained* fall of more than `drop` below the running maximum.

    A candidate must stay below the threshold for `persist_s` of samples before
    it fires, and the event is timed at the start of that run so lead time is
    measured from the first crossing, not the deepest point.

    `phase_at` is an optional callable `t_s -> phase`. When it is None the gate is
    not applied and every event carries `phase=None`; an ungated measurement is a
    weaker claim and callers must surface it as such.
    """
    series = [s for s in samples if s.clean] if use_clean_only else list(samples)
    if len(series) < 2:
        return []
    interval = _sample_interval(samples)
    expected = window_s / interval if interval > 0 else 0.0

    fractions: list[tuple[AreaSample, float]] = []
    for i, s in enumerate(series):
        window = [o for o in series[: i + 1] if o.t_s >= s.t_s - window_s]
        wmax = max((o.normalised_area for o in window), default=0.0)
        fractions.append((s, (wmax - s.normalised_area) / wmax if wmax > 0 else 0.0))

    events: list[DropEvent] = []
    last_fired = -1e9
    i = 0
    while i < len(fractions):
        if fractions[i][1] <= drop:
            i += 1
            continue
        j = i
        while j + 1 < len(fractions) and fractions[j + 1][1] > drop:
            j += 1
        start, end = fractions[i][0], fractions[j][0]
        sustained = end.t_s - start.t_s
        covered = sum(1 for o in series if start.t_s - window_s <= o.t_s <= start.t_s)
        before = sum(1 for o in series if start.t_s - window_s <= o.t_s < start.t_s)
        after = sum(1 for o in series if end.t_s < o.t_s <= end.t_s + window_s)
        dense = (expected <= 0 or covered >= min_window_coverage * expected)
        flanked = before >= min_flank_samples and after >= min_flank_samples
        if sustained >= persist_s and dense and flanked:
            phase = phase_at(start.t_s) if phase_at is not None else None
            gated_out = phase_at is not None and phase not in gate_phases
            if not gated_out and start.t_s - last_fired >= rate_limit_s:
                last_fired = start.t_s
                run = fractions[i : j + 1]
                deepest = max(run, key=lambda p: p[1])
                window = [o for o in series if start.t_s - window_s <= o.t_s <= start.t_s]
                events.append(DropEvent(
                    t_s=start.t_s,
                    drop_fraction=fractions[i][1],
                    max_drop_fraction=deepest[1],
                    window_max=max((o.normalised_area for o in window), default=0.0),
                    area=start.normalised_area,
                    sustained_s=sustained,
                    phase=phase,
                    used_clean_frames=use_clean_only,
                ))
        i = j + 1
    return events


def _sample_interval(samples: list[AreaSample]) -> float:
    """Median spacing of the *sampled* series, including frames later rejected."""
    if len(samples) < 2:
        return 0.0
    gaps = sorted(b.t_s - a.t_s for a, b in zip(samples, samples[1:]) if b.t_s > a.t_s)
    return gaps[len(gaps) // 2] if gaps else 0.0


def compare_raw_and_filtered(samples: list[AreaSample], **kw) -> dict:
    """Both traces side by side, so the effect of the occlusion filter is visible."""
    raw = drop_events(samples, use_clean_only=False, **kw)
    filtered = drop_events(samples, use_clean_only=True, **kw)
    n_clean = sum(1 for s in samples if s.clean)
    return {
        "n_samples": len(samples),
        "n_clean": n_clean,
        "n_occluded": len(samples) - n_clean,
        "raw_events": raw,
        "filtered_events": filtered,
        "suppressed_by_filter": len(raw) - len(filtered),
    }


def deepest_drop(samples: list[AreaSample], window_s: float = DEFAULT_WINDOW_S,
                 use_clean_only: bool = True) -> float:
    """Largest fractional drop in the series — used to rank candidate videos."""
    series = [s for s in samples if s.clean] if use_clean_only else list(samples)
    worst = 0.0
    for i, s in enumerate(series):
        window = [o for o in series[: i + 1] if o.t_s >= s.t_s - window_s]
        wmax = max((o.normalised_area for o in window), default=0.0)
        if wmax > 0:
            worst = max(worst, (wmax - s.normalised_area) / wmax)
    return worst


DEFAULT_MERGE_GAP_S = 60.0
DEFAULT_RECOVERY_FRACTION = 0.85


@dataclass(frozen=True)
class DropEpisode:
    """One constriction, however many times the rule re-crossed during it.

    A running maximum decays towards a sustained low. When a pupil constricts and
    stays constricted, the fraction slips back under the threshold and the
    continuing decline re-crosses it, so one episode is reported as several
    events. Measured on case_742: five events between 426 s and 610 s while the
    window maximum fell 0.45 -> 0.34 and only 11% of samples recovered to 85% of
    baseline. That is one episode, not five alerts.
    """

    start_s: float
    end_s: float
    max_drop_fraction: float
    baseline: float
    n_events: int
    events: tuple[DropEvent, ...]

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def observation(self) -> str:
        gate = "ungated" if self.events[0].phase is None else f"during {self.events[0].phase}"
        return (
            f"measured pupil area down to {self.max_drop_fraction * 100:.0f}% below baseline, "
            f"sustained {self.duration_s:.0f} s ({gate})"
        )

    def as_dict(self) -> dict:
        return {
            "start_s": round(self.start_s, 2), "end_s": round(self.end_s, 2),
            "duration_s": round(self.duration_s, 2),
            "max_drop_fraction": round(self.max_drop_fraction, 4),
            "baseline": round(self.baseline, 6), "n_events_merged": self.n_events,
        }


def group_episodes(
    events: list[DropEvent],
    samples: list[AreaSample],
    merge_gap_s: float = DEFAULT_MERGE_GAP_S,
    recovery_fraction: float = DEFAULT_RECOVERY_FRACTION,
) -> list[DropEpisode]:
    """Merge events that belong to one uninterrupted constriction.

    Two consecutive events join when the gap between them is short *and* the
    pupil does not recover during it. Recovery is the deciding test: if the area
    climbs back to `recovery_fraction` of the episode's baseline, the next drop
    is a genuinely new constriction and starts a new episode.
    """
    if not events:
        return []
    clean = [s for s in samples if s.clean]
    groups: list[list[DropEvent]] = [[events[0]]]
    for previous, current in zip(events, events[1:]):
        prev_end = previous.t_s + previous.sustained_s
        baseline = groups[-1][0].window_max
        recovered = any(
            prev_end <= s.t_s <= current.t_s and baseline > 0
            and s.normalised_area >= recovery_fraction * baseline
            for s in clean
        )
        if current.t_s - prev_end <= merge_gap_s and not recovered:
            groups[-1].append(current)
        else:
            groups.append([current])
    return [
        DropEpisode(
            start_s=g[0].t_s,
            end_s=g[-1].t_s + g[-1].sustained_s,
            max_drop_fraction=max(e.max_drop_fraction for e in g),
            baseline=g[0].window_max,
            n_events=len(g),
            events=tuple(g),
        )
        for g in groups
    ]
