"""Two rankings over the same screening results, for two different jobs.

**Confirmation** asks which cases are most worth spending SAM 2 time on, so it
ranks by drop depth alone: the deepest measured drops are the ones whose truth
matters most.

**Demo** asks which cases will read honestly on screen, so it ranks by
`drop depth x clean fraction`. A deep drop measured over sparse data is a poor
thing to show: the dashboard would spend most of the case with its markers greyed
out, and the one event would rest on a handful of frames.

The two lists are allowed to differ and are labelled separately. Neither is
confirmed until SAM 2 has run.

Pure arithmetic, no network (CLAUDE.md rule 2).
"""
from __future__ import annotations

N_SUBSET = 38          # size of the Cataract-1K pupil-reaction subset


def combined_score(row: dict) -> float:
    """Drop depth weighted by how much of the case was measurable."""
    return float(row.get("deepest_filtered_drop", 0.0)) * float(row.get("clean_fraction", 0.0))


def rank_for_demo(results: list[dict], top: int = 2, require_event: bool = True) -> list[dict]:
    """Cases to play in the mock demo, best first.

    A case with no event is excluded by default: there would be nothing to show.
    """
    pool = [r for r in results if not require_event or r.get("n_filtered_events", 0) > 0]
    ranked = sorted(pool, key=lambda r: (-combined_score(r), r.get("video", "")))
    return [dict(r, combined_score=round(combined_score(r), 4)) for r in ranked[:top]]


def rank_for_confirmation(results: list[dict], top: int = 5) -> list[dict]:
    """Cases to send to SAM 2, deepest measured drop first."""
    ranked = sorted(results, key=lambda r: (-float(r.get("deepest_filtered_drop", 0.0)),
                                            r.get("video", "")))
    return [dict(r, combined_score=round(combined_score(r), 4)) for r in ranked[:top]]


def demo_label(n_screened: int, n_subset: int = N_SUBSET) -> str:
    missing = n_subset - n_screened
    scope = f"{n_screened} of {n_subset}" if missing > 0 else str(n_subset)
    tail = f" ({missing} not retrieved)" if missing > 0 else ""
    return (f"demo cases selected by drop depth x clean fraction across {scope}{tail}; "
            f"measured pupil-area decrease, ungated; SAM 2 confirmation pending")


def confirmation_label(n_screened: int, n_confirmed: int = 0,
                       n_subset: int = N_SUBSET) -> str:
    missing = n_subset - n_screened
    scope = f"{n_screened} of {n_subset}" if missing > 0 else str(n_subset)
    tail = f" ({missing} not retrieved)" if missing > 0 else ""
    state = (f"SAM 2-confirmed on {n_confirmed}" if n_confirmed
             else "SAM 2 confirmation pending")
    return f"ranked by classical screener across {scope}{tail}; {state}"
