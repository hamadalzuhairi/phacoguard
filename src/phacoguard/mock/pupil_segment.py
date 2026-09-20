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

import json

from phacoguard.data.phase_map import PUPIL_GATE_STEPS, TRANSITION
from phacoguard.detectors.pupil_measured import (
    DropEpisode, drop_events, group_episodes, read_area_csv,
)
from phacoguard.mock.cataract1k import PhaseInterval
from phacoguard.mock import cataract1k
from phacoguard.mock.sources import MarkerResult, MarkerStatus, MockEvent, Provenance
from phacoguard.mock.timeline import (
    MEASURED_BANNER, NO_PHACO_LABELS, NO_PHASE_LABELS, NO_RADIAL_SOURCE,
    CaseTimeline, StateChange,
)
from phacoguard.pipeline.fusion import RiskFusion


def load_predictions(path: Path) -> dict:
    """Predicted step intervals from the quick phase model."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    d["phase_intervals"] = [
        PhaseInterval(raw_name=i["phase"], phase=i["phase"],
                      start_s=float(i["start_s"]), end_s=float(i["end_s"]))
        for i in d["intervals"] if i["phase"] != TRANSITION
    ]
    return d


def _crossing(spans, threshold_s: float):
    """Wall-clock time at which accumulated phaco reaches `threshold_s`."""
    acc = 0.0
    for span in sorted(spans, key=lambda s: s.start_s):
        remaining = threshold_s - acc
        if remaining <= span.duration_s:
            return span.start_s + remaining
        acc += span.duration_s
    return None


def build(case_id: str, video_path: Path, trace_csv: Path, fps: float = 25.0,
          duration_s: float | None = None, predictions: dict | None = None,
          distribution: dict | None = None) -> CaseTimeline:
    """A segment driven by the measured pupil-area decrease.

    With `predictions`, the step list, step timeline and phaco timer come from the
    quick phase model and the pupil marker is gated to the predicted steps. The
    ungated result is kept alongside, because a gate built on 87%-accurate
    predictions can hide a real constriction as easily as it can suppress a false
    one, and the claim must stay visible either way.
    """
    samples = read_area_csv(trace_csv)
    all_events = drop_events(samples)
    episodes_ungated = group_episodes(all_events, samples)
    span = duration_s if duration_s is not None else (samples[-1].t_s if samples else 0.0)

    intervals: list[PhaseInterval] = []
    episodes = episodes_ungated
    gated_note = ""
    if predictions:
        intervals = predictions["phase_intervals"]

        def predicted_step(t: float) -> str:
            for iv in intervals:
                if iv.start_s <= t < iv.end_s:
                    return iv.phase
            return TRANSITION

        episodes = [e for e in episodes_ungated
                    if predicted_step(e.start_s) in PUPIL_GATE_STEPS]
        dropped = len(episodes_ungated) - len(episodes)
        gated_note = (f"gated to predicted {', '.join(PUPIL_GATE_STEPS)}; "
                      f"{len(episodes)} of {len(episodes_ungated)} episode(s) retained"
                      + (f", {dropped} outside those steps" if dropped else ""))

    provenance = Provenance(
        dataset=cataract1k.DATASET,
        licence=cataract1k.DATASET_LICENCE,
        case_id=case_id,
        source_files=(f"{trace_csv.name}",),
        derivation=(
            "measured pupil area from classical red/dark segmentation; "
            + (gated_note if predictions else "ungated (no phase labels for this case)")
            + "; SAM 2 confirmation pending"
        ),
    )
    events = [
        MockEvent(type="pupil_constriction", t_s=ep.start_s,
                  observation=ep.observation, provenance=provenance)
        for ep in episodes
    ]

    pupil_note = (f"{len(episodes)} episode(s) from "
                  f"{sum(e.n_events for e in episodes)} crossing(s); measured, not annotated")
    if predictions:
        pupil_note += f"; {gated_note}"
    markers = {
        "pupil_constriction": MarkerResult(
            marker="pupil_constriction", status=MarkerStatus.MEASURED, events=events,
            note=pupil_note,
        ),
        "radial_folds": MarkerResult(
            marker="radial_folds", status=MarkerStatus.NO_LABELLED_SOURCE,
            note=NO_RADIAL_SOURCE),
        "prolonged_phaco": MarkerResult(
            marker="prolonged_phaco",
            status=MarkerStatus.MEASURED if predictions else MarkerStatus.NO_LABELLED_SOURCE,
            note=("phaco duration from predicted steps" if predictions else NO_PHACO_LABELS)),
    }

    phaco_s = sum(i.duration_s for i in intervals if i.phase == "phaco")
    percentile = 0.0
    phaco_events: list[MockEvent] = []
    if predictions and distribution:
        values = sorted(distribution["durations_s"].values())
        percentile = 100.0 * sum(1 for v in values if v < phaco_s) / max(len(values), 1)
        phaco_prov = Provenance(
            dataset=cataract1k.DATASET, licence=cataract1k.DATASET_LICENCE, case_id=case_id,
            source_files=(f"{case_id}.json",),
            derivation=("accumulated phaco time from predicted steps, against the empirical "
                        f"percentile of {distribution['n_cases']} labelled cases"),
        )
        phaco_spans = [i for i in intervals if i.phase == "phaco"]
        for key in ("p85", "p95"):
            t = _crossing(phaco_spans, distribution[key])
            if t is not None:
                phaco_events.append(MockEvent(
                    type="prolonged_phaco", t_s=t,
                    observation=(f"predicted phaco time past the {key[1:]}th percentile "
                                 f"({distribution[key]:.0f} s) of {distribution['n_cases']} "
                                 f"labelled cases"),
                    provenance=phaco_prov))
        markers["prolonged_phaco"] = MarkerResult(
            marker="prolonged_phaco", status=MarkerStatus.MEASURED, events=phaco_events,
            note=(f"{phaco_s:.0f}s of predicted phaco (percentile {percentile:.0f}); "
                  f"{len(phaco_events)} crossing(s)"))

    ordered = sorted([*events, *phaco_events], key=lambda e: e.t_s)
    states = [StateChange(0.0, "routine")]
    counts = {"pupil_constriction": 0, "prolonged_phaco": 0}
    for e in ordered:
        counts[e.type] = counts.get(e.type, 0) + 1
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
        intervals=intervals,
        markers=markers,
        events=ordered,
        states=states,
        phaco_duration_s=phaco_s,
        phaco_percentile=percentile,
        distribution=(distribution or {}) if predictions else {},
        # The phaco panel needs both predicted steps and the reference
        # distribution; predictions alone give a duration with nothing to
        # compare it against.
        unavailable=_unavailable(predictions, distribution),
        episodes=episodes,
        caption=caption(case_id, "measured", predictions),
        banner=_banner(predictions),
        footer_source=_footer(case_id, predictions),
    )


def quick_model_phrase(predictions: dict) -> str:
    return (f"phases predicted by a quick model (n={predictions['n_train_cases']} training cases, "
            f"test accuracy {predictions['test_frame_accuracy']*100:.0f}%); full model pending")


def caption(case_id: str, pupil_driver: str, predictions: dict | None = None) -> str:
    """One line naming the dataset, case, licence and what drives each indicator."""
    if predictions:
        merged = predictions.get("merged_for_bar") or {}
        note = ("  |  not yet distinguished: "
                + ", ".join(f"{a.replace('_',' ')} vs {b.replace('_',' ')}"
                            for a, b in merged.items())) if merged else ""
        return (f"Cataract-1K {case_id} (CC BY 4.0)  |  {quick_model_phrase(predictions)}  |  "
                f"pupil: measured area decrease, gated to predicted steps  |  "
                f"phaco: predicted steps  |  radial folds: access pending{note}")
    return (f"Cataract-1K {case_id} (CC BY 4.0)  |  phase: no labels  |  "
            f"pupil: measured area decrease (not annotated)  |  phaco: no labels  |  "
            f"radial folds: access pending")


def _unavailable(predictions: dict | None, distribution: dict | None) -> dict:
    out: dict[str, str] = {"radial": NO_RADIAL_SOURCE}
    if not predictions:
        out["phase"] = NO_PHASE_LABELS
    if not (predictions and distribution):
        out["phaco"] = NO_PHACO_LABELS
    return out


def _banner(predictions: dict | None) -> str:
    if predictions:
        return ("Pupil marker is a measurement, not an expert annotation; "
                + quick_model_phrase(predictions) + ".")
    return MEASURED_BANNER


def _footer(case_id: str, predictions: dict | None) -> str:
    base = (f"Source: {cataract1k.DATASET} {case_id} ({cataract1k.DATASET_LICENCE}). "
            f"Pupil area measured by classical red/dark segmentation")
    if predictions:
        return (base + f"; steps are model predictions, not expert labels "
                       f"(n={predictions['n_train_cases']} training cases, test accuracy "
                       f"{predictions['test_frame_accuracy']*100:.0f}%).")
    return base + "; this case has no phase labels and no pupil-reaction annotation."


def episode_lines(episodes: list[DropEpisode]) -> list[str]:
    """Short lines for the event log, one per episode."""
    return [
        f"{int(ep.start_s) // 60:02d}:{int(ep.start_s) % 60:02d}  "
        f"{ep.max_drop_fraction * 100:.0f}% for {ep.duration_s:.0f}s"
        for ep in episodes
    ]
