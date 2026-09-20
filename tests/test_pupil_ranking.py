"""Tests for the demo and confirmation rankings."""
from __future__ import annotations

import pytest

from phacoguard.detectors.pupil_ranking import (
    combined_score, confirmation_label, demo_label, rank_for_confirmation, rank_for_demo,
)


def row(video, drop, clean, events=1):
    return {"video": video, "deepest_filtered_drop": drop,
            "clean_fraction": clean, "n_filtered_events": events}


RESULTS = [
    row("case_716", 0.890, 0.37),      # deepest drop, sparsest data
    row("case_742", 0.687, 0.84),      # best combined
    row("case_800", 0.642, 0.79),
    row("case_8114", 0.776, 0.62),
    row("case_769", 0.789, 0.56),
]


def test_combined_score_is_drop_times_clean():
    assert combined_score(row("x", 0.5, 0.5)) == pytest.approx(0.25)


def test_demo_prefers_measurable_cases_over_the_deepest_drop():
    picked = [r["video"] for r in rank_for_demo(RESULTS, top=2)]
    assert picked == ["case_742", "case_800"]
    assert "case_716" not in picked, "89% over 37% clean is a poor thing to show"


def test_confirmation_still_ranks_by_depth():
    picked = [r["video"] for r in rank_for_confirmation(RESULTS, top=2)]
    assert picked == ["case_716", "case_769"]


def test_the_two_rankings_are_allowed_to_differ():
    demo = {r["video"] for r in rank_for_demo(RESULTS, top=2)}
    conf = {r["video"] for r in rank_for_confirmation(RESULTS, top=2)}
    assert demo != conf


def test_cases_without_an_event_are_excluded_from_the_demo():
    results = RESULTS + [row("case_silent", 0.99, 0.99, events=0)]
    assert "case_silent" not in [r["video"] for r in rank_for_demo(results, top=3)]


def test_silent_cases_may_still_be_confirmed():
    results = RESULTS + [row("case_silent", 0.99, 0.99, events=0)]
    assert rank_for_confirmation(results, top=1)[0]["video"] == "case_silent"


def test_ranking_is_deterministic_on_ties():
    tied = [row("case_b", 0.5, 0.5), row("case_a", 0.5, 0.5)]
    assert [r["video"] for r in rank_for_demo(tied, top=2)] == ["case_a", "case_b"]


def test_score_is_attached_to_each_returned_row():
    assert rank_for_demo(RESULTS, top=1)[0]["combined_score"] == pytest.approx(0.5771, abs=1e-3)


# --- labels ------------------------------------------------------------------

def test_demo_label_names_the_metric_and_the_missing_videos():
    label = demo_label(31)
    assert "drop depth x clean fraction" in label
    assert "31 of 38" in label and "7 not retrieved" in label
    assert "SAM 2 confirmation pending" in label


def test_confirmation_label_is_distinct_from_the_demo_label():
    assert confirmation_label(31) != demo_label(31)
    assert "classical screener" in confirmation_label(31)


def test_labels_drop_the_caveat_once_the_subset_is_complete():
    assert "not retrieved" not in demo_label(38)
    assert "not retrieved" not in confirmation_label(38)


def test_confirmation_label_reports_confirmed_count():
    assert "SAM 2-confirmed on 3" in confirmation_label(31, n_confirmed=3)
