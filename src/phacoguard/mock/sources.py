"""Label-driven event sources for the section 2a mock demo.

Each source either produces events whose timing comes from an expert dataset
label, or reports that it has no labelled source. There is no third option: a
marker is never placed on the timeline by our own reading of the video.

`MarkerStatus.NO_LABELLED_SOURCE` is deliberately distinct from "the marker did
not fire". The renderer shows the two differently so that a missing data source
can never be read as a negative finding.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from phacoguard.mock import cataract1k
from phacoguard.mock.cataract1k import PhaseInterval


class MarkerStatus(str, Enum):
    LABELLED = "labelled"          # timing comes from an expert dataset label
    MEASURED = "measured"          # timing comes from our own measurement, not a label
    NO_LABELLED_SOURCE = "no_labelled_source"


@dataclass(frozen=True)
class Provenance:
    """Where a displayed moment came from. Rendered on screen next to the event."""

    dataset: str
    licence: str
    case_id: str
    source_files: tuple[str, ...]
    derivation: str

    def as_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "licence": self.licence,
            "case_id": self.case_id,
            "source_files": list(self.source_files),
            "derivation": self.derivation,
        }


@dataclass(frozen=True)
class MockEvent:
    """One indicator firing.

    `confidence` is None throughout section 2a. No model has run, so there is no
    confidence to report, and inventing one would misrepresent the demo.
    """

    type: str
    t_s: float
    observation: str
    provenance: Provenance
    confidence: float | None = None

    def as_dict(self) -> dict:
        return {
            "type": self.type,
            "t_s": round(self.t_s, 2),
            "confidence": self.confidence,
            "explanation": self.observation,
            "provenance": self.provenance.as_dict(),
        }


@dataclass
class MarkerResult:
    """What one marker contributes to a case."""

    marker: str
    status: MarkerStatus
    events: list[MockEvent] = field(default_factory=list)
    note: str = ""


def phase_timeline(intervals: list[PhaseInterval], case_id: str) -> list[dict]:
    """The phase bar: expert labels passed through unchanged."""
    return [
        {
            "phase": i.phase,
            "raw_label": i.raw_name,
            "start_s": round(i.start_s, 2),
            "end_s": round(i.end_s, 2),
            "source": f"{case_id}_annotations_phases.csv",
        }
        for i in intervals
    ]


def phaco_duration_distribution(annotations_root: Path, pmap: dict) -> dict:
    """Empirical phaco-duration percentiles over every labelled case.

    The distribution is a property of the expert labels, not a threshold we
    chose, so a case crossing its 85th percentile is a label-derived event.
    """
    durations: dict[str, float] = {}
    for case_dir in cataract1k.find_cases(annotations_root):
        intervals, _ = cataract1k.load_case(case_dir, pmap)
        durations[case_dir.name] = cataract1k.phaco_duration_s(intervals)
    if len(durations) < 2:
        raise ValueError(
            f"need at least 2 labelled cases to form a distribution, found {len(durations)} "
            f"under {annotations_root}"
        )
    values = sorted(durations.values())
    return {
        "n_cases": len(values),
        "durations_s": durations,
        "p50": _percentile(values, 50),
        "p85": _percentile(values, 85),
        "p95": _percentile(values, 95),
        "source": "labelled phaco intervals of all annotated Cataract-1K cases",
    }


def prolonged_phaco(
    intervals: list[PhaseInterval], case_id: str, distribution: dict
) -> MarkerResult:
    """Fires when accumulated labelled phaco time crosses the 85th and 95th percentiles."""
    phaco = [i for i in intervals if i.phase == "phaco"]
    if not phaco:
        return MarkerResult(
            marker="prolonged_phaco",
            status=MarkerStatus.LABELLED,
            note="no phaco phase labelled in this case",
        )
    prov = Provenance(
        dataset=cataract1k.DATASET,
        licence=cataract1k.DATASET_LICENCE,
        case_id=case_id,
        source_files=(f"{case_id}_annotations_phases.csv", f"{case_id}_video.csv"),
        derivation=(
            f"accumulated labelled phaco time crossing the empirical percentile of "
            f"{distribution['n_cases']} labelled cases"
        ),
    )
    events = []
    for pct in ("p85", "p95"):
        t = crossing_time(phaco, distribution[pct])
        if t is not None:
            events.append(
                MockEvent(
                    type="prolonged_phaco",
                    t_s=t,
                    observation=(
                        f"phaco time past the {pct[1:]}th percentile "
                        f"({distribution[pct]:.0f} s) of {distribution['n_cases']} labelled cases"
                    ),
                    provenance=prov,
                )
            )
    return MarkerResult("prolonged_phaco", MarkerStatus.LABELLED, events)


def pupil_constriction(case_id: str) -> MarkerResult:
    """No labelled source in the current Cataract-1K release.

    The pupil-reaction subset (syn53395402) is 38 mp4 files and no annotation
    file: the label is at case level, with no interval saying when the pupil
    constricts. Choosing a moment ourselves is exactly what section 2a forbids,
    so this marker reports that it has no labelled source.
    """
    return MarkerResult(
        marker="pupil_constriction",
        status=MarkerStatus.NO_LABELLED_SOURCE,
        note=(
            "Cataract-1K pupil-reaction subset carries a case-level flag only; "
            "no labelled interval exists to drive this indicator"
        ),
    )


def radial_folds(case_id: str) -> MarkerResult:
    """No labelled source until Tongren-Zonular-Video access is granted."""
    return MarkerResult(
        marker="radial_folds",
        status=MarkerStatus.NO_LABELLED_SOURCE,
        note="Tongren-Zonular-Video is gated (manual approval); no clip labels available yet",
    )


def crossing_time(phaco_intervals: list[PhaseInterval], threshold_s: float) -> float | None:
    """Wall-clock time at which accumulated phaco time first reaches `threshold_s`.

    Returns None when the case never accumulates that much phaco time.
    """
    if threshold_s <= 0:
        return phaco_intervals[0].start_s if phaco_intervals else None
    accumulated = 0.0
    for interval in sorted(phaco_intervals, key=lambda i: i.start_s):
        remaining = threshold_s - accumulated
        if remaining <= interval.duration_s:
            return interval.start_s + remaining
        accumulated += interval.duration_s
    return None


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile, matching numpy's default."""
    if not sorted_values:
        raise ValueError("empty distribution")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    k = (len(sorted_values) - 1) * pct / 100.0
    lo, hi = int(k), min(int(k) + 1, len(sorted_values) - 1)
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo))


def median_case(distribution: dict) -> str:
    """Case whose labelled phaco duration is closest to the median — the routine segment."""
    med = statistics.median(distribution["durations_s"].values())
    return min(distribution["durations_s"], key=lambda c: abs(distribution["durations_s"][c] - med))


def highest_case(distribution: dict) -> str:
    """Case with the longest labelled phaco duration — the complex segment."""
    return max(distribution["durations_s"], key=lambda c: distribution["durations_s"][c])
