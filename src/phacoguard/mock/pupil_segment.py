"""Build a demo segment from a pupil-reaction case.

These cases carry a case-level clinician flag and nothing else: no phase labels,
no interval annotation. So the segment shows a *measured* pupil-area decrease and
states plainly, in each empty panel, why the other panels have no data source.

Constrictions are shown as **episodes**, not as one alert per threshold crossing.
A 30 s running maximum decays towards a sustained low, so a pupil that constricts
and stays constricted re-crosses the threshold repeatedly: case_742 produced
seven events for two episodes. Seven alerts would misrepresent one constriction
as seven.

No network access (CLAUDE.md rule 2).
"""
from __future__ import annotations

from pathlib import Path

from phacoguard.detectors.pupil_measured import (
    DropEpisode, drop_events, group_episodes, read_area_csv,
)
from phacoguard.mock import cataract1k
from phacoguard.mock.sources import MarkerResult, MarkerStatus, MockEvent, Provenance
from phacoguard.mock.timeline import (
    MEASURED_BANNER, NO_PHACO_LABELS, NO_PHASE_LABELS, NO_RADIAL_SOURCE,
    CaseTimeline, StateChange,
)
from phacoguard.pipeline.fusion import RiskFusion


def build(case_id: str, video_path: Path, trace_csv: Path, fps: float = 25.0,
          duration_s: float | None = None) -> CaseTimeline:
    """A segment driven by the measured pupil-area decrease."""
    samples = read_area_csv(trace_csv)
    episodes = group_episodes(drop_events(samples), samples)
    span = duration_s if duration_s is not None else (samples[-1].t_s if samples else 0.0)

    provenance = Provenance(
        dataset=cataract1k.DATASET,
        licence=cataract1k.DATASET_LICENCE,
        case_id=case_id,
        source_files=(f"{trace_csv.name}",),
        derivation=(
            "measured pupil area from classical red/dark segmentation, ungated "
            "(no phase labels for this case); SAM 2 confirmation pending"
        ),
    )
    events = [
        MockEvent(type="pupil_constriction", t_s=ep.start_s,
                  observation=ep.observation, provenance=provenance)
        for ep in episodes
    ]

    markers = {
        "pupil_constriction": MarkerResult(
            marker="pupil_constriction", status=MarkerStatus.MEASURED, events=events,
            note=f"{len(episodes)} episode(s) from {sum(e.n_events for e in episodes)} threshold "
                 f"crossing(s); measured, not annotated",
        ),
        "radial_folds": MarkerResult(
            marker="radial_folds", status=MarkerStatus.NO_LABELLED_SOURCE,
            note=NO_RADIAL_SOURCE),
        "prolonged_phaco": MarkerResult(
            marker="prolonged_phaco", status=MarkerStatus.NO_LABELLED_SOURCE,
            note=NO_PHACO_LABELS),
    }

    states = [StateChange(0.0, "routine")]
    counts = {"pupil_constriction": 0}
    for e in events:
        counts["pupil_constriction"] += 1
        state = RiskFusion.state(counts)
        if state != states[-1].state:
            states.append(StateChange(e.t_s, state))

    return CaseTimeline(
        case_id=case_id,
        dataset=cataract1k.DATASET,
        licence=cataract1k.DATASET_LICENCE,
        video_path=video_path,
        fps=fps,
        duration_s=span,
        intervals=[],
        markers=markers,
        events=events,
        states=states,
        phaco_duration_s=0.0,
        phaco_percentile=0.0,
        distribution={},
        unavailable={"phase": NO_PHASE_LABELS, "phaco": NO_PHACO_LABELS},
        episodes=episodes,
        caption=caption(case_id, "measured"),
        banner=MEASURED_BANNER,
        footer_source=(
            f"Source: {cataract1k.DATASET} {case_id} ({cataract1k.DATASET_LICENCE}). "
            f"Pupil area measured by classical red/dark segmentation; this case has no "
            f"phase labels and no pupil-reaction annotation."
        ),
    )


def caption(case_id: str, pupil_driver: str) -> str:
    """One line naming the dataset, case, licence and what drives each indicator."""
    pupil = {
        "measured": "pupil: measured area decrease (not annotated)",
        "none": "pupil: no annotation",
    }[pupil_driver]
    if pupil_driver == "measured":
        return (f"Cataract-1K {case_id} (CC BY 4.0)  |  phase: no labels  |  "
                f"{pupil}  |  phaco: no labels  |  radial folds: access pending")
    return (f"Cataract-1K {case_id} (CC BY 4.0)  |  phase: expert labels  |  "
            f"{pupil}  |  phaco: expert phase labels  |  radial folds: access pending")


def episode_lines(episodes: list[DropEpisode]) -> list[str]:
    """Short lines for the event log, one per episode."""
    return [
        f"{int(ep.start_s) // 60:02d}:{int(ep.start_s) % 60:02d}  "
        f"{ep.max_drop_fraction * 100:.0f}% for {ep.duration_s:.0f}s"
        for ep in episodes
    ]
