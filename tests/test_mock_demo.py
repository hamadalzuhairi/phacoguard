"""Tests for the section 2a label-driven mock demo.

These run without any dataset: the annotation files are synthesised in the
shape the Cataract-1K authors' own reader expects.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from phacoguard.mock import cataract1k, sources, timeline as tl  # noqa: E402
from phacoguard.render import dashboard  # noqa: E402

PMAP = yaml.safe_load(
    open(Path(__file__).resolve().parents[1] / "configs/phase_map.yaml", encoding="utf-8")
)
FPS = 30.0


def write_case(root: Path, case_id: str, rows: list[tuple[str, float, float]]) -> Path:
    """Create `case_<id>/` with the two CSVs, taking phase bounds in seconds."""
    d = root / case_id
    d.mkdir(parents=True)
    (d / f"{case_id}_video.csv").write_text(
        "caseId,width,height,frames,fps\n" f"{case_id},1024,768,9999,{FPS}\n", encoding="utf-8"
    )
    lines = ["caseId,comment,frame,endFrame,sec,endSec"]
    for name, start_s, end_s in rows:
        lines.append(
            f"{case_id},{name},{int(start_s * FPS)},{int(end_s * FPS)},{start_s},{end_s}"
        )
    (d / f"{case_id}_annotations_phases.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return d


def simple_case(root: Path, case_id: str, phaco_s: float) -> Path:
    return write_case(
        root,
        case_id,
        [("Incision", 0, 10), ("Capsulorhexis", 10, 40), ("Hydrodissection", 40, 50),
         ("Phacoemulsification", 50, 50 + phaco_s),
         ("Irrigation/Aspiration", 50 + phaco_s, 60 + phaco_s),
         ("Lens Implantation", 60 + phaco_s, 75 + phaco_s)],
    )


# --- readers -----------------------------------------------------------------

def test_reads_fps_and_intervals(tmp_path):
    d = simple_case(tmp_path, "case_1001", phaco_s=120)
    intervals, fps = cataract1k.load_case(d, PMAP)
    assert fps == FPS
    assert [i.phase for i in intervals] == [
        "incision", "capsulorhexis", "hydrodissection", "phaco", "cortex_removal", "iol_insertion"
    ]
    assert cataract1k.phaco_duration_s(intervals) == pytest.approx(120, abs=0.1)


def test_unmapped_phase_name_raises(tmp_path):
    d = write_case(tmp_path, "case_1002", [("Sculpting", 0, 5)])
    with pytest.raises(KeyError, match="Unmapped phase"):
        cataract1k.load_case(d, PMAP)


def test_malformed_row_raises(tmp_path):
    d = simple_case(tmp_path, "case_1003", phaco_s=60)
    p = d / "case_1003_annotations_phases.csv"
    p.write_text(p.read_text(encoding="utf-8") + "case_1003,Incision,notanumber,5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-numeric"):
        cataract1k.load_case(d, PMAP)


def test_end_before_start_raises(tmp_path):
    d = write_case(tmp_path, "case_1004", [("Incision", 10, 10)])
    p = d / "case_1004_annotations_phases.csv"
    p.write_text("caseId,comment,frame,endFrame\ncase_1004,Incision,300,120\n", encoding="utf-8")
    with pytest.raises(ValueError, match="precedes start"):
        cataract1k.load_case(d, PMAP)


# --- distribution and crossing ----------------------------------------------

def test_distribution_over_labelled_cases(tmp_path):
    for i, phaco in enumerate([60, 90, 120, 150, 600]):
        simple_case(tmp_path, f"case_{2000 + i}", phaco_s=phaco)
    dist = sources.phaco_duration_distribution(tmp_path, PMAP)
    assert dist["n_cases"] == 5
    assert dist["p50"] == pytest.approx(120, abs=1)
    assert dist["p85"] < dist["p95"] <= 600
    assert sources.highest_case(dist) == "case_2004"


def test_single_case_cannot_form_distribution(tmp_path):
    simple_case(tmp_path, "case_2100", phaco_s=60)
    with pytest.raises(ValueError, match="at least 2 labelled cases"):
        sources.phaco_duration_distribution(tmp_path, PMAP)


def test_crossing_time_accumulates_across_intervals(tmp_path):
    d = write_case(
        tmp_path, "case_2200",
        [("Phacoemulsification", 50, 80), ("Irrigation/Aspiration", 80, 90),
         ("Phacoemulsification", 90, 140)],
    )
    intervals, _ = cataract1k.load_case(d, PMAP)
    phaco = [i for i in intervals if i.phase == "phaco"]
    # 30 s accumulates by the end of the first interval; the next 10 s land in the second.
    assert sources.crossing_time(phaco, 30) == pytest.approx(80, abs=0.1)
    assert sources.crossing_time(phaco, 40) == pytest.approx(100, abs=0.1)
    assert sources.crossing_time(phaco, 999) is None


# --- markers -----------------------------------------------------------------

def test_markers_without_labels_report_no_source():
    assert sources.pupil_constriction("case_1").status is sources.MarkerStatus.NO_LABELLED_SOURCE
    assert sources.radial_folds("case_1").status is sources.MarkerStatus.NO_LABELLED_SOURCE
    assert sources.pupil_constriction("case_1").events == []


def test_prolonged_phaco_events_carry_provenance(tmp_path):
    for i, phaco in enumerate([30, 40, 50, 60, 900]):
        simple_case(tmp_path, f"case_{3000 + i}", phaco_s=phaco)
    dist = sources.phaco_duration_distribution(tmp_path, PMAP)
    intervals, _ = cataract1k.load_case(tmp_path / "case_3004", PMAP)
    result = sources.prolonged_phaco(intervals, "case_3004", dist)
    assert result.status is sources.MarkerStatus.LABELLED
    assert len(result.events) == 2  # p85 and p95 both crossed
    for e in result.events:
        assert e.confidence is None, "no model has run, so no confidence may be reported"
        assert e.provenance.dataset == "cataract1k"
        assert e.provenance.licence == "CC BY 4.0"
        assert "case_3004_annotations_phases.csv" in e.provenance.source_files


def test_short_case_fires_nothing(tmp_path):
    for i, phaco in enumerate([30, 40, 50, 60, 900]):
        simple_case(tmp_path, f"case_{3100 + i}", phaco_s=phaco)
    dist = sources.phaco_duration_distribution(tmp_path, PMAP)
    intervals, _ = cataract1k.load_case(tmp_path / "case_3100", PMAP)
    assert sources.prolonged_phaco(intervals, "case_3100", dist).events == []


# --- timeline ----------------------------------------------------------------

def _timeline(tmp_path, phaco_s: float) -> tl.CaseTimeline:
    for i, p in enumerate([30, 40, 50, 60, 900]):
        simple_case(tmp_path, f"case_{4000 + i}", phaco_s=p)
    target = simple_case(tmp_path, "case_4999", phaco_s=phaco_s)
    dist = sources.phaco_duration_distribution(tmp_path, PMAP)
    intervals, _ = cataract1k.load_case(target, PMAP)
    markers = {
        "pupil_constriction": sources.pupil_constriction("case_4999"),
        "radial_folds": sources.radial_folds("case_4999"),
        "prolonged_phaco": sources.prolonged_phaco(intervals, "case_4999", dist),
    }
    return tl.build(target, tmp_path / "missing.mp4", PMAP, dist, markers)


def test_timeline_matches_schema_fields(tmp_path):
    d = _timeline(tmp_path, phaco_s=1200).as_dict()
    for key in ("case_id", "phases", "events", "phaco_duration_s", "phaco_percentile",
                "final_state", "model_versions"):
        assert key in d
    assert d["mode"] == "mock_label_driven"
    assert d["model_versions"]["phase"] is None, "no trained model may be claimed in mock mode"
    assert d["markers"]["pupil_constriction"]["status"] == "no_labelled_source"


def test_rate_limit_suppresses_events_within_30s(tmp_path):
    t = _timeline(tmp_path, phaco_s=1200)
    times = [e.t_s for e in t.events if e.type == "prolonged_phaco"]
    assert all(b - a >= 30.0 for a, b in zip(times, times[1:]))


def test_state_rises_and_phase_lookup(tmp_path):
    t = _timeline(tmp_path, phaco_s=1200)
    assert t.state_at(0) == "routine"
    assert t.final_state in ("rising", "high")
    assert t.phase_at(20) == "capsulorhexis"
    assert t.phase_at(60) == "phaco"


def test_routine_case_stays_routine(tmp_path):
    t = _timeline(tmp_path, phaco_s=35)
    assert t.events == []
    assert t.final_state == "routine"


# --- rule 4 ------------------------------------------------------------------

def test_dashboard_text_is_observational():
    dashboard.assert_observational(dashboard.ui_strings())


def test_observational_guard_catches_instructions():
    with pytest.raises(AssertionError):
        dashboard.assert_observational(["Consider slowing the phaco"])
    with pytest.raises(AssertionError):
        dashboard.assert_observational(["Surgical guidance panel"])


def test_event_observations_are_observational(tmp_path):
    t = _timeline(tmp_path, phaco_s=1200)
    dashboard.assert_observational([e.observation for e in t.events])


def test_frame_renders_without_video(tmp_path):
    t = _timeline(tmp_path, phaco_s=1200)
    frame = dashboard.render_frame(t, None, t_s=120.0)
    assert frame.shape == (dashboard.H, dashboard.W, 3)


# --- dataset vocabulary ------------------------------------------------------

REAL_CATARACT1K_PHASES = {
    "Incision", "Viscoelastic", "Capsulorhexis", "Hydrodissection", "Phacoemulsification",
    "Irrigation/Aspiration", "Capsule Pulishing", "Lens Implantation", "Lens positioning",
    "Viscoelastic_Suction", "Anterior_Chamber Flushing", "Tonifying/Antibiotics",
}


def test_phase_map_covers_the_real_vocabulary():
    """The 12 names observed across all 56 published annotation files, and no others.

    Verified 20 Sep 2026. "Capsule Pulishing" is the dataset's own spelling.
    """
    assert set(PMAP["cataract1k"]) == REAL_CATARACT1K_PHASES


def test_every_mapped_phase_is_a_phacoguard_phase():
    from phacoguard.data.phase_map import PHASES
    assert set(PMAP["cataract1k"].values()) <= set(PHASES)


# --- segment availability and captions ---------------------------------------

def test_phase_segment_states_why_pupil_is_unavailable(tmp_path):
    t = _timeline(tmp_path, phaco_s=1200)
    assert t.unavailable["pupil"] == tl.NO_PUPIL_ANNOTATION
    assert t.unavailable["radial"] == tl.NO_RADIAL_SOURCE
    assert "phase: expert labels" in t.caption and "CC BY 4.0" in t.caption


def test_unavailable_reasons_are_observational():
    dashboard.assert_observational([
        tl.NO_PHASE_LABELS, tl.NO_PHACO_LABELS, tl.NO_PUPIL_ANNOTATION,
        tl.NO_RADIAL_SOURCE, tl.MEASURED_BANNER,
    ])


def test_a_measured_segment_does_not_claim_expert_labels(tmp_path):
    """The footer must describe its own segment, not the label-driven ones."""
    from phacoguard.mock import pupil_segment
    csv_path = tmp_path / "trace.csv"
    rows = ["frame_index,t_s,pupil_px,limbus_px,normalised_pupil_area,circularity,"
            "fill_ratio,specular_fraction,instrument_overlap,mode,limbus_r,clean,reject_reason"]
    for i, t in enumerate(range(0, 400, 2)):
        area = 0.50 if t < 120 else 0.22
        rows.append(f"{i},{t},{int(area*10000)},10000,{area},0.9,0.95,0.0,0.0,red,100,1,")
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    seg = pupil_segment.build("case_742", tmp_path / "v.mp4", csv_path)
    assert seg.banner == tl.MEASURED_BANNER
    # The banner may mention expert annotation only to deny it.
    assert "not an expert annotation" in seg.banner.lower()
    assert "driven by expert dataset labels" not in seg.banner.lower()
    assert seg.banner != tl.MOCK_BANNER
    assert "measured" in seg.footer_source.lower()
    assert "no phase labels" in seg.footer_source
    assert seg.unavailable["phase"] == tl.NO_PHASE_LABELS
    assert seg.unavailable["phaco"] == tl.NO_PHACO_LABELS
    assert seg.markers["pupil_constriction"].status.value == "measured"
    assert seg.episodes and seg.events
    dashboard.assert_observational([seg.caption, seg.banner, seg.footer_source])


def test_a_measured_segment_renders(tmp_path):
    from phacoguard.mock import pupil_segment
    csv_path = tmp_path / "trace.csv"
    rows = ["frame_index,t_s,pupil_px,limbus_px,normalised_pupil_area,circularity,"
            "fill_ratio,specular_fraction,instrument_overlap,mode,limbus_r,clean,reject_reason"]
    for i, t in enumerate(range(0, 400, 2)):
        area = 0.50 if t < 120 else 0.22
        rows.append(f"{i},{t},{int(area*10000)},10000,{area},0.9,0.95,0.0,0.0,red,100,1,")
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    seg = pupil_segment.build("case_742", tmp_path / "v.mp4", csv_path)
    frame = dashboard.render_frame(seg, None, t_s=200.0)
    assert frame.shape == (dashboard.H, dashboard.W, 3)


# --- surgical step taxonomy --------------------------------------------------

from phacoguard.data.phase_map import (  # noqa: E402
    PHASES, PUPIL_GATE_STEPS, SURGICAL_STEPS, TRANSITION, step_index, to_phacoguard,
)


def test_eleven_classes_ten_of_them_steps():
    assert len(SURGICAL_STEPS) == 10
    assert PHASES == [*SURGICAL_STEPS, TRANSITION]
    assert TRANSITION not in SURGICAL_STEPS


def test_steps_are_in_fixed_surgical_order():
    assert SURGICAL_STEPS == [
        "incision", "ovd_injection", "capsulorhexis", "hydrodissection", "phaco",
        "cortex_removal", "capsule_polishing", "iol_insertion", "ovd_removal",
        "wound_closure",
    ]
    assert [step_index(s) for s in SURGICAL_STEPS] == list(range(1, 11))
    assert step_index(TRANSITION) == 0


def test_every_cataract1k_label_maps_to_exactly_one_step():
    for label in REAL_CATARACT1K_PHASES:
        assert to_phacoguard("cataract1k", label, PMAP) in SURGICAL_STEPS


def test_no_label_may_map_to_transition():
    bad = {"cataract1k": {"Something": TRANSITION}}
    with pytest.raises(ValueError, match="not a surgical step"):
        to_phacoguard("cataract1k", "Something", bad)


def test_unmapped_label_raises_rather_than_defaulting():
    with pytest.raises(KeyError, match="Unmapped phase"):
        to_phacoguard("cataract1k", "Sculpting", PMAP)


def test_pupil_gate_covers_phaco_cortex_and_polishing():
    assert PUPIL_GATE_STEPS == ("phaco", "cortex_removal", "capsule_polishing")


def test_transition_is_the_label_between_intervals(tmp_path):
    d = write_case(tmp_path, "case_9001", [("Incision", 0, 10), ("Phacoemulsification", 40, 90)])
    intervals, _ = cataract1k.load_case(d, PMAP)
    t = tl.build(d, tmp_path / "v.mp4", PMAP,
                 {"n_cases": 2, "p50": 1.0, "p85": 1.0, "p95": 1.0, "durations_s": {"a": 1, "b": 2}},
                 {})
    assert t.phase_at(20) == TRANSITION
    assert t.phase_at(50) == "phaco"
    assert all(r["step"] != TRANSITION for r in t.step_status(50)), "transition is never a step row"


def test_step_status_and_times(tmp_path):
    d = simple_case(tmp_path, "case_9002", phaco_s=100)
    t = tl.build(d, tmp_path / "v.mp4", PMAP,
                 {"n_cases": 2, "p50": 1.0, "p85": 1.0, "p95": 1.0, "durations_s": {"a": 1, "b": 2}},
                 {})
    rows = {r["step"]: r for r in t.step_status(60)}
    assert [r["index"] for r in t.step_status(60)] == list(range(1, 11))
    assert rows["incision"]["status"] == "completed"
    assert rows["phaco"]["status"] == "in progress"
    assert rows["wound_closure"]["status"] == "not performed"
    assert rows["capsulorhexis"]["seconds"] == pytest.approx(30, abs=0.5)


def test_a_recurring_step_never_shows_time_while_upcoming(tmp_path):
    """OVD is injected twice; it must not read 'upcoming' with time already spent."""
    d = write_case(tmp_path, "case_9003", [
        ("Viscoelastic", 0, 8), ("Phacoemulsification", 10, 60), ("Viscoelastic", 70, 78)])
    t = tl.build(d, tmp_path / "v.mp4", PMAP,
                 {"n_cases": 2, "p50": 1.0, "p85": 1.0, "p95": 1.0, "durations_s": {"a": 1, "b": 2}},
                 {})
    for row in t.step_status(30):
        if row["status"] == "upcoming":
            assert row["seconds"] == 0, f"{row['step']} is upcoming but shows time spent"
    assert {r["step"]: r["status"] for r in t.step_status(30)}["ovd_injection"] == "completed"


def test_phase_changes_are_logged_in_order(tmp_path):
    d = simple_case(tmp_path, "case_9004", phaco_s=100)
    t = tl.build(d, tmp_path / "v.mp4", PMAP,
                 {"n_cases": 2, "p50": 1.0, "p85": 1.0, "p95": 1.0, "durations_s": {"a": 1, "b": 2}},
                 {})
    changes = t.phase_changes()
    # simple_case has no Viscoelastic interval, so ovd_injection is absent here.
    assert [c[1] for c in changes][:3] == ["incision", "capsulorhexis", "hydrodissection"]
    assert changes == sorted(changes, key=lambda c: c[0])
    assert dashboard.step_line(5, "phaco") == "Step 5: phaco started"


# --- predicted phases --------------------------------------------------------

def _trace_csv(tmp_path):
    rows = ["frame_index,t_s,pupil_px,limbus_px,normalised_pupil_area,circularity,"
            "fill_ratio,specular_fraction,instrument_overlap,mode,limbus_r,clean,reject_reason"]
    for i, t in enumerate(range(0, 400, 2)):
        area = 0.50 if t < 120 else 0.22
        rows.append(f"{i},{t},{int(area*10000)},10000,{area},0.9,0.95,0.0,0.0,red,100,1,")
    p = tmp_path / "trace.csv"
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return p


def _predictions(tmp_path, phase="phaco"):
    import json
    p = tmp_path / "pred.json"
    p.write_text(json.dumps({
        "case_id": "case_742", "source": "quick phase model", "n_train_cases": 18,
        "test_frame_accuracy": 0.876, "merged_for_bar": {"ovd_injection": "capsulorhexis"},
        "fps": 0.5,
        "intervals": [{"phase": "incision", "start_s": 0, "end_s": 40},
                      {"phase": phase, "start_s": 100, "end_s": 400}],
    }), encoding="utf-8")
    return p


def test_a_predicted_segment_cannot_claim_expert_phase_labels(tmp_path):
    from phacoguard.mock import pupil_segment
    pred = pupil_segment.load_predictions(_predictions(tmp_path))
    seg = pupil_segment.build("case_742", tmp_path / "v.mp4", _trace_csv(tmp_path),
                              predictions=pred)
    blob = " ".join([seg.caption, seg.banner, seg.footer_source]).lower()
    # "expert labels" may appear only inside an explicit denial.
    assert "phase: expert labels" not in blob
    assert "derive from the dataset's expert phase labels" not in blob
    assert blob.count("expert") == blob.count("not expert") + blob.count("not an expert")
    assert "predicted by a quick model" in blob
    assert "n=18 training cases" in blob and "88%" in blob
    assert "full model pending" in blob
    assert "not expert labels" in seg.footer_source.lower()
    dashboard.assert_observational([seg.caption, seg.banner, seg.footer_source])


def test_predicted_segment_states_which_steps_are_not_distinguished(tmp_path):
    from phacoguard.mock import pupil_segment
    pred = pupil_segment.load_predictions(_predictions(tmp_path))
    seg = pupil_segment.build("case_742", tmp_path / "v.mp4", _trace_csv(tmp_path),
                              predictions=pred)
    assert "not yet distinguished" in seg.caption
    assert "ovd injection vs capsulorhexis" in seg.caption


def test_predicted_segment_gates_the_pupil_marker_and_keeps_the_ungated_count(tmp_path):
    from phacoguard.mock import pupil_segment
    inside = pupil_segment.build(
        "case_742", tmp_path / "v.mp4", _trace_csv(tmp_path),
        predictions=pupil_segment.load_predictions(_predictions(tmp_path, phase="phaco")))
    outside = pupil_segment.build(
        "case_742", tmp_path / "v.mp4", _trace_csv(tmp_path),
        predictions=pupil_segment.load_predictions(_predictions(tmp_path, phase="incision")))
    assert inside.episodes, "a drop inside a gated step survives"
    assert not outside.episodes, "a drop outside the gated steps is suppressed"
    note = outside.markers["pupil_constriction"].note
    assert "of 1 episode(s) retained" in note and "outside those steps" in note


def test_predicted_segment_has_a_step_list_and_no_phase_reason(tmp_path):
    from phacoguard.mock import pupil_segment
    pred = pupil_segment.load_predictions(_predictions(tmp_path))
    seg = pupil_segment.build("case_742", tmp_path / "v.mp4", _trace_csv(tmp_path),
                              predictions=pred)
    assert "phase" not in seg.unavailable, "predicted steps mean the phase bar has a source"
    assert seg.intervals and {r["step"] for r in seg.step_status(200)} <= set(SURGICAL_STEPS)
    assert dashboard.render_frame(seg, None, t_s=200.0).shape == (dashboard.H, dashboard.W, 3)
