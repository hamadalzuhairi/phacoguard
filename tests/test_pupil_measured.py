"""Tests for the measured pupil-area decrease detector."""
from __future__ import annotations

import pytest

from phacoguard.detectors.pupil_measured import (
    AreaSample, compare_raw_and_filtered, deepest_drop, drop_events, read_area_csv,
)


def series(pairs, clean=True) -> list[AreaSample]:
    return [AreaSample(t, a, clean) for t, a in pairs]


def flat_then_drop(level=0.40, low=0.28, switch=40, end=100):
    return series([(t, level) for t in range(0, switch, 2)]
                  + [(t, low) for t in range(switch, end, 2)])


# --- threshold ---------------------------------------------------------------

def test_no_drop_on_flat_series():
    assert drop_events(series([(t, 0.30) for t in range(0, 120, 2)])) == []


def test_fires_on_a_sustained_30_percent_drop():
    ev = drop_events(flat_then_drop())
    assert ev
    assert ev[0].t_s == 40, "the event is timed at the first crossing, for lead time"
    assert ev[0].max_drop_fraction == pytest.approx(0.30, abs=0.01)


def test_does_not_fire_just_under_threshold():
    assert drop_events(flat_then_drop(low=0.33)) == []      # 17.5% < 20%


def test_drop_outside_the_window_is_not_counted():
    s = series([(0, 0.50)] + [(t, 0.30) for t in range(40, 120, 2)])
    assert drop_events(s) == []


# --- persistence -------------------------------------------------------------

def test_brief_dip_does_not_fire():
    """A 4 s dip is shorter than the 10 s persistence rule."""
    s = series([(t, 0.40) for t in range(0, 40, 2)]
               + [(40, 0.28), (42, 0.28)]
               + [(t, 0.40) for t in range(44, 90, 2)])
    assert drop_events(s) == []


def test_dip_lasting_ten_seconds_fires():
    s = series([(t, 0.40) for t in range(0, 40, 2)]
               + [(t, 0.28) for t in range(40, 52, 2)]
               + [(t, 0.40) for t in range(52, 90, 2)])
    ev = drop_events(s)
    assert ev and ev[0].sustained_s >= 10.0


# --- occlusion filtering -----------------------------------------------------

def test_occluded_frames_are_excluded():
    """An instrument shrinks the visible pupil; those frames must not fire it."""
    s = ([AreaSample(t, 0.40, True) for t in range(0, 40, 2)]
         + [AreaSample(t, 0.20, False, "circularity 0.5") for t in range(40, 60, 2)]
         + [AreaSample(t, 0.40, True) for t in range(60, 90, 2)])
    assert drop_events(s, use_clean_only=True) == []
    assert drop_events(s, use_clean_only=False), "unfiltered, the occlusion reads as a drop"


def test_real_drop_survives_the_filter():
    s = ([AreaSample(t, 0.40, True) for t in range(0, 40, 2)]
         + [AreaSample(t, 0.26, True) for t in range(40, 80, 2)])
    assert drop_events(s, use_clean_only=True)


def test_compare_raw_and_filtered_reports_the_difference():
    s = ([AreaSample(t, 0.40, True) for t in range(0, 40, 2)]
         + [AreaSample(t, 0.18, False, "specular 0.４") for t in range(40, 60, 2)]
         + [AreaSample(t, 0.40, True) for t in range(60, 90, 2)])
    c = compare_raw_and_filtered(s)
    assert c["n_occluded"] == 10
    assert c["suppressed_by_filter"] >= 1
    assert len(c["filtered_events"]) < len(c["raw_events"])


# --- gating ------------------------------------------------------------------

def test_ungated_events_carry_no_phase():
    ev = drop_events(flat_then_drop())
    assert ev and ev[0].phase is None


def test_phase_gate_still_available_for_the_trained_model_rerun():
    s = flat_then_drop()
    assert drop_events(s, phase_at=lambda t: "capsulorhexis") == []
    assert drop_events(s, phase_at=lambda t: "phaco")


def test_rate_limit_holds_between_events():
    s = series([(t, 0.40) for t in range(0, 40, 2)]
               + [(t, 0.20) for t in range(40, 200, 2)])
    ts = [e.t_s for e in drop_events(s, rate_limit_s=30.0)]
    assert all(b - a >= 30.0 for a, b in zip(ts, ts[1:]))


# --- wording and IO ----------------------------------------------------------

def test_observation_says_measured_and_ungated():
    text = drop_events(flat_then_drop())[0].observation.lower()
    assert "measured" in text and "ungated" in text
    assert "annotat" not in text


def test_deepest_drop_ranks_series():
    assert deepest_drop(flat_then_drop(low=0.20)) > deepest_drop(flat_then_drop(low=0.36))


def test_reads_tracker_csv_with_clean_column(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text(
        "frame_index,t_s,pupil_px,limbus_px,normalised_pupil_area,circularity,"
        "specular_fraction,instrument_overlap,clean,reject_reason\n"
        "0,0.0,100,1000,0.1,0.95,0.0,0.0,1,\n"
        "1,0.5,0,1000,0.0,0.0,0.0,0.0,0,no pupil mask\n"
        "2,1.0,60,1000,0.06,0.55,0.0,0.0,0,circularity 0.55\n",
        encoding="utf-8",
    )
    s = read_area_csv(p)
    assert [x.t_s for x in s] == [0.0, 1.0]
    assert s[1].clean is False and "circularity" in s[1].reject_reason


# --- clean-sample density ----------------------------------------------------

def test_sparse_window_does_not_fire():
    """A drop whose 30 s window is mostly occluded must not be trusted."""
    s = ([AreaSample(t, 0.40, True) for t in range(0, 20, 2)]
         + [AreaSample(t, 0.40, False, "specular") for t in range(20, 60, 2)]
         + [AreaSample(t, 0.26, True) for t in range(60, 100, 2)])
    assert drop_events(s) == [], "window coverage is too low to support a drop"


def test_well_covered_window_still_fires():
    s = ([AreaSample(t, 0.40, True) for t in range(0, 40, 2)]
         + [AreaSample(t, 0.26, True) for t in range(40, 100, 2)])
    assert drop_events(s)


def test_drop_without_enough_clean_samples_before_does_not_fire():
    s = ([AreaSample(t, 0.40, True) for t in range(0, 6, 2)]
         + [AreaSample(t, 0.26, True) for t in range(6, 60, 2)])
    assert drop_events(s) == [], "fewer than 5 clean samples precede the drop"


def test_drop_at_the_very_end_does_not_fire():
    s = ([AreaSample(t, 0.40, True) for t in range(0, 40, 2)]
         + [AreaSample(t, 0.26, True) for t in range(40, 54, 2)])
    assert drop_events(s) == [], "fewer than 5 clean samples follow the drop"


# --- episode grouping --------------------------------------------------------

from phacoguard.detectors.pupil_measured import group_episodes  # noqa: E402


def _stepwise_decline() -> list[AreaSample]:
    """A pupil that constricts in steps and never recovers.

    Each plateau lets the 30 s running maximum catch down to the new level, so
    the next step crosses the threshold again. This is the case_742 pattern: one
    constriction, reported by the rule as several events.
    """
    levels = [(0, 60, 0.60), (60, 120, 0.45), (120, 180, 0.35),
              (180, 240, 0.27), (240, 320, 0.21)]
    return [AreaSample(float(t), v, True)
            for lo, hi, v in levels for t in range(lo, hi, 2)]


def test_a_single_event_is_one_episode():
    s = flat_then_drop()
    ev = drop_events(s)
    eps = group_episodes(ev, s)
    assert len(eps) == 1 and eps[0].n_events == 1


def test_sustained_constriction_is_one_episode_not_many_alerts():
    """The running maximum decays, so a continuing fall re-crosses the threshold."""
    s = _stepwise_decline()
    ev = drop_events(s)
    assert len(ev) > 1, "the rule re-fires during one long decline"
    eps = group_episodes(ev, s)
    assert len(eps) == 1, "but it is one constriction"
    assert eps[0].n_events == len(ev)
    assert eps[0].duration_s > 60


def test_recovery_between_drops_starts_a_new_episode():
    s = ([AreaSample(t, 0.60, True) for t in range(0, 60, 2)]
         + [AreaSample(t, 0.35, True) for t in range(60, 110, 2)]
         + [AreaSample(t, 0.60, True) for t in range(110, 200, 2)]
         + [AreaSample(t, 0.35, True) for t in range(200, 260, 2)])
    eps = group_episodes(drop_events(s), s)
    assert len(eps) == 2, "the pupil recovered in between, so these are two constrictions"


def test_a_long_quiet_gap_starts_a_new_episode():
    s = ([AreaSample(t, 0.60, True) for t in range(0, 60, 2)]
         + [AreaSample(t, 0.35, True) for t in range(60, 100, 2)]
         + [AreaSample(t, 0.42, True) for t in range(100, 400, 2)]
         + [AreaSample(t, 0.20, True) for t in range(400, 460, 2)])
    eps = group_episodes(drop_events(s), s, merge_gap_s=60.0)
    assert len(eps) >= 2


def test_episode_reports_duration_and_deepest_drop():
    s = _stepwise_decline()
    ep = group_episodes(drop_events(s), s)[0]
    assert ep.duration_s == pytest.approx(ep.end_s - ep.start_s)
    assert ep.max_drop_fraction == max(e.max_drop_fraction for e in ep.events)
    assert "sustained" in ep.observation and "measured" in ep.observation


def test_episode_wording_is_observational():
    from phacoguard.render import dashboard
    s = _stepwise_decline()
    dashboard.assert_observational([e.observation for e in group_episodes(drop_events(s), s)])


def test_no_events_gives_no_episodes():
    assert group_episodes([], []) == []
