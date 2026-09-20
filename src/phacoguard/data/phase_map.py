"""Map dataset-specific phase names to PhacoGuard's surgical steps.

Ten steps in fixed surgical order, plus `transition`.

`transition` is **not a surgical step**. It is what the system reports when the
current moment falls between labelled intervals, and it appears only as the
current-phase label — never as a row in the step list. An earlier taxonomy
collapsed OVD injection, capsule polishing, OVD removal and wound closure into a
single "idle" class, which both hid four real steps and made "idle" the most
common label in the dataset.

No dataset label may map to `transition`: an unmapped label raises, because a
label quietly falling into `transition` would silently delete a step from a case.
"""
from __future__ import annotations

import yaml

#: The ten surgical steps, in the order they occur.
SURGICAL_STEPS = [
    "incision",
    "ovd_injection",
    "capsulorhexis",
    "hydrodissection",
    "phaco",
    "cortex_removal",
    "capsule_polishing",
    "iol_insertion",
    "ovd_removal",
    "wound_closure",
]

TRANSITION = "transition"

#: Everything the phase model can emit: the ten steps plus `transition`.
PHASES = [*SURGICAL_STEPS, TRANSITION]

#: Steps during which a pupil-area decrease is gated (detectors.yaml).
PUPIL_GATE_STEPS = ("phaco", "cortex_removal", "capsule_polishing")

#: Step during which radial folds are gated.
RADIAL_FOLD_GATE_STEPS = ("capsulorhexis",)


def load_map(path="configs/phase_map.yaml"):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def to_phacoguard(dataset: str, label: str, pmap: dict) -> str:
    """Translate one dataset label into a surgical step.

    Raises on an unmapped label rather than defaulting, so a vocabulary change in
    a dataset surfaces as an error instead of a missing step.
    """
    table = pmap.get(dataset) or {}
    out = table.get(label)
    if out is None:
        raise KeyError(
            f"Unmapped phase {label!r} for dataset {dataset!r}; "
            f"add it to configs/phase_map.yaml under {dataset}"
        )
    if out == TRANSITION:
        raise ValueError(
            f"{dataset}:{label!r} maps to {TRANSITION!r}, which is not a surgical step. "
            f"Map it to one of: {', '.join(SURGICAL_STEPS)}"
        )
    if out not in SURGICAL_STEPS:
        raise ValueError(f"{dataset}:{label!r} maps to unknown step {out!r}")
    return out


def step_index(step: str) -> int:
    """1-based position in the surgical order; 0 for `transition`."""
    return SURGICAL_STEPS.index(step) + 1 if step in SURGICAL_STEPS else 0
