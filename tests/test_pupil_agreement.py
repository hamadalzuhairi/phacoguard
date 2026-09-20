"""Tests for screener-vs-SAM 2 agreement."""
from __future__ import annotations

import pytest

from phacoguard.detectors.pupil_agreement import compare
from phacoguard.detectors.pupil_measured import AreaSample


def trace(pairs, clean=True):
    return [AreaSample(t, a, clean) for t, a in pairs]


def flat_then_drop(level=0.40, low=0.26, switch=40, end=110, offset=0.0):
    return trace([(t, level + offset) for t in range(0, switch, 2)]
                 + [(t, low + offset) for t in range(switch, end, 2)])


def test_identical_traces_agree():
    a = flat_then_drop()
    ag = compare(a, list(a))
    assert ag.pearson_r == pytest.approx(1.0, abs=1e-6)
    assert ag.matched and not ag.classical_only and not ag.sam2_only
    assert ag.verdict == "good agreement"


def test_offset_but_correlated_traces_still_match_events():
    """A constant offset changes absolute area but not the relative drop."""
    a = flat_then_drop()
    b = flat_then_drop(offset=0.05)
    ag = compare(a, b)
    assert ag.pearson_r > 0.99
    assert ag.matched


def test_event_present_in_one_trace_only_is_reported():
    a = flat_then_drop()
    b = trace([(t, 0.40) for t in range(0, 110, 2)])     # no drop at all
    ag = compare(a, b)
    assert ag.classical_only and not ag.matched
    assert ag.verdict != "good agreement"


def test_uncorrelated_traces_are_poor():
    a = trace([(t, 0.40 + 0.01 * (t % 5)) for t in range(0, 110, 2)])
    b = trace([(t, 0.40 - 0.01 * (t % 7)) for t in range(0, 110, 2)])
    ag = compare(a, b)
    assert ag.verdict in ("poor agreement", "partial agreement")


def test_events_within_tolerance_are_matched():
    a = flat_then_drop(switch=40)
    b = flat_then_drop(switch=46)          # 6 s later, inside the 15 s tolerance
    ag = compare(a, b)
    assert ag.matched and ag.matched[0][2] <= 15.0


def test_events_outside_tolerance_are_not_matched():
    a = flat_then_drop(switch=40, end=200)
    b = flat_then_drop(switch=120, end=200)
    ag = compare(a, b, tolerance_s=15.0)
    assert not ag.matched
    assert ag.classical_only and ag.sam2_only


def test_too_little_overlap_is_not_judged():
    a = trace([(0.0, 0.4), (2.0, 0.4)])
    b = trace([(0.0, 0.4), (2.0, 0.4)])
    assert compare(a, b).verdict == "insufficient overlap to judge"


def test_occluded_samples_are_excluded_from_correlation():
    a = flat_then_drop()
    b = [AreaSample(s.t_s, 99.0, False, "occluded") for s in a]
    ag = compare(a, b)
    assert ag.n_paired == 0, "occluded samples must not enter the correlation"
