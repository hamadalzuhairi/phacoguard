"""Assemble a label-driven case into the timeline JSON of `configs/timeline_schema.json`.

The timeline is the only thing the record writer ever sees (ARCHITECTURE.md §3,
component 7), so it carries the provenance of every event with it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from phacoguard.data.phase_map import SURGICAL_STEPS, TRANSITION, step_index
from phacoguard.mock import cataract1k
from phacoguard.mock.cataract1k import PhaseInterval
from phacoguard.mock.sources import MarkerResult, MarkerStatus, MockEvent, phase_timeline
from phacoguard.pipeline.fusion import RiskFusion

MARKERS = ("pupil_constriction", "radial_folds", "prolonged_phaco")

MOCK_BANNER = (
    "Indicators driven by expert dataset labels; model outputs replace them as training completes."
)

# A panel with no data source states why, rather than sitting blank or greyed.
# A blank panel reads as "nothing found"; these read as "nothing to look with".
NO_PHASE_LABELS = "phase bar unavailable: no phase labels for this case"
NO_PHACO_LABELS = "phaco timing unavailable: no phase labels for this case"
NO_PUPIL_ANNOTATION = "pupil marker unavailable: no pupil-reaction annotation for this case"
NO_RADIAL_SOURCE = "radial folds unavailable: Tongren-Zonular-Video access pending"

# The footer must describe the segment it is under. A label-driven segment and a
# measured one make different claims, and one banner for both would be false on
# whichever it did not describe.
MEASURED_BANNER = (
    "Pupil marker is a measurement, not an expert annotation; SAM 2 confirmation pending."
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
    # Panel key -> the reason it has no data source, shown in the panel itself.
    unavailable: dict = field(default_factory=dict)
    # Pupil constriction episodes, for segments driven by the measured marker.
    episodes: list = field(default_factory=list)
    # One line naming dataset, case, licence and what drives each indicator.
    caption: str = ""
    # Footer text for this segment. Defaults describe a label-driven case.
    banner: str = MOCK_BANNER
    footer_source: str = ""

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
        """The current step, or `transition` when between labelled intervals."""
        for i in self.intervals:
            if i.start_s <= t_s < i.end_s:
                return i.phase
        return TRANSITION

    def step_status(self, t_s: float) -> list[dict]:
        """One row per surgical step, in fixed order, with status and time spent.

        `transition` never appears here: it is a current-phase label, not a step.
        """
        rows = []
        for step in SURGICAL_STEPS:
            spans = [i for i in self.intervals if i.phase == step]
            spent = sum(max(0.0, min(t_s, i.end_s) - i.start_s) for i in spans)
            # Some steps recur: OVD is injected at the start and again before the
            # IOL. "completed" therefore means completed so far, and a recurring
            # step flips back to "in progress" when it resumes. Testing for a
            # pending interval first would label a step "upcoming" while showing
            # time already spent in it, which reads as a contradiction.
            if not spans:
                status = "not performed"
            elif any(i.start_s <= t_s < i.end_s for i in spans):
                status = "in progress"
            elif any(i.end_s <= t_s for i in spans):
                status = "completed"
            else:
                status = "upcoming"
            rows.append({"index": step_index(step), "step": step,
                         "status": status, "seconds": spent,
                         "total_seconds": sum(i.duration_s for i in spans)})
        return rows

    def phase_changes(self) -> list[tuple[float, str]]:
        """Every change of step, as (time, step), for the event log."""
        out: list[tuple[float, str]] = []
        for i in sorted(self.intervals, key=lambda i: i.start_s):
            if not out or out[-1][1] != i.phase:
                out.append((i.start_s, i.phase))
        return out

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
            } if self.distribution else None,
            "unavailable": dict(self.unavailable),
            "caption": self.caption,
            "episodes": [e.as_dict() for e in self.episodes],
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
        unavailable={"pupil": NO_PUPIL_ANNOTATION, "radial": NO_RADIAL_SOURCE},
        caption=(f"Cataract-1K {case_id} (CC BY 4.0)  |  phase: expert labels  |  "
                 f"pupil: no annotation  |  phaco: expert phase labels  |  "
                 f"radial folds: access pending"),
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
