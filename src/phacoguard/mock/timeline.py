"""Assemble a label-driven case into the timeline JSON of `configs/timeline_schema.json`.

The timeline is the only thing the record writer ever sees (ARCHITECTURE.md §3,
component 7), so it carries the provenance of every event with it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from phacoguard.mock import cataract1k
from phacoguard.mock.cataract1k import PhaseInterval
from phacoguard.mock.sources import MarkerResult, MarkerStatus, MockEvent, phase_timeline
from phacoguard.pipeline.fusion import RiskFusion

MARKERS = ("pupil_constriction", "radial_folds", "prolonged_phaco")

MOCK_BANNER = (
    "Indicators driven by expert dataset labels; model outputs replace them as training completes."
)


@dataclass
class StateChange:
    t_s: float
    state: str


@dataclass
class CaseTimeline:
    case_id: str
    dataset: str
    licence: str
    video_path: Path
    fps: float
    duration_s: float
    intervals: list[PhaseInterval]
    markers: dict[str, MarkerResult]
    events: list[MockEvent]
    states: list[StateChange]
    phaco_duration_s: float
    phaco_percentile: float
    distribution: dict

    @property
    def final_state(self) -> str:
        return self.states[-1].state if self.states else "routine"

    def state_at(self, t_s: float) -> str:
        state = "routine"
        for change in self.states:
            if change.t_s <= t_s:
                state = change.state
            else:
                break
        return state

    def phase_at(self, t_s: float) -> str:
        for i in self.intervals:
            if i.start_s <= t_s < i.end_s:
                return i.phase
        return "idle"

    def as_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "dataset": self.dataset,
            "dataset_licence": self.licence,
            "dataset_citation": cataract1k.DATASET_CITATION,
            "mode": "mock_label_driven",
            "mock_banner": MOCK_BANNER,
            "phases": phase_timeline(self.intervals, self.case_id),
            "events": [e.as_dict() for e in self.events],
            "markers": {
                name: {"status": r.status.value, "note": r.note, "n_events": len(r.events)}
                for name, r in self.markers.items()
            },
            "phaco_duration_s": round(self.phaco_duration_s, 2),
            "phaco_percentile": round(self.phaco_percentile, 1),
            "phaco_distribution": {
                "n_cases": self.distribution["n_cases"],
                "p50": round(self.distribution["p50"], 1),
                "p85": round(self.distribution["p85"], 1),
                "p95": round(self.distribution["p95"], 1),
            },
            "state_changes": [{"t_s": round(s.t_s, 2), "state": s.state} for s in self.states],
            "final_state": self.final_state,
            "model_versions": {
                "seg": None,
                "phase": None,
                "detectors": None,
                "fusion": "rule-based fusion over label-driven events (no trained model)",
            },
        }


def build(
    case_dir: Path,
    video_path: Path,
    pmap: dict,
    distribution: dict,
    marker_results: dict[str, MarkerResult],
    rate_limit_s: float = 30.0,
) -> CaseTimeline:
    """Build one case timeline from its expert labels and the marker results."""
    case_id = cataract1k.case_dir_id(case_dir)
    intervals, fps = cataract1k.load_case(case_dir, pmap)
    phaco_s = cataract1k.phaco_duration_s(intervals)

    events = _rate_limited(marker_results, rate_limit_s)
    states = _state_changes(events)

    return CaseTimeline(
        case_id=case_id,
        dataset=cataract1k.DATASET,
        licence=cataract1k.DATASET_LICENCE,
        video_path=video_path,
        fps=fps,
        duration_s=max((i.end_s for i in intervals), default=0.0),
        intervals=intervals,
        markers=marker_results,
        events=events,
        states=states,
        phaco_duration_s=phaco_s,
        phaco_percentile=_percentile_of(phaco_s, distribution),
        distribution=distribution,
    )


def _rate_limited(marker_results: dict[str, MarkerResult], rate_limit_s: float) -> list[MockEvent]:
    """Apply the 30 s per-marker rate limit (detectors.yaml) in time order."""
    fusion = RiskFusion(rate_limit_s=rate_limit_s)
    ordered = sorted(
        (e for r in marker_results.values() for e in r.events), key=lambda e: e.t_s
    )
    return [e for e in ordered if fusion.allow(e.type, e.t_s)]


def _state_changes(events: list[MockEvent]) -> list[StateChange]:
    """Risk band over time, recomputed from the events accumulated so far."""
    counts = {m: 0 for m in MARKERS}
    changes = [StateChange(0.0, "routine")]
    for event in events:
        counts[event.type] = counts.get(event.type, 0) + 1
        state = RiskFusion.state(counts)
        if state != changes[-1].state:
            changes.append(StateChange(event.t_s, state))
    return changes


def _percentile_of(value: float, distribution: dict) -> float:
    """Where this case's phaco duration sits in the labelled distribution (0-100)."""
    values = sorted(distribution["durations_s"].values())
    if not values:
        return 0.0
    below = sum(1 for v in values if v < value)
    return 100.0 * below / len(values)
