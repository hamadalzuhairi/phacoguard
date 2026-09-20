# Working prototype demonstration

*Hackathon deliverable (rule 3.3): live demo or recorded video. Plan: live demo with a recorded fallback.*

## What the judges see (3 minutes)

1. **Setup (0:00-0:20).** Laptop with PhacoGuard running offline; second monitor (or split screen) shows the dashboard. Network visibly disabled (airplane mode on screen).
2. **Routine case (0:20-1:00).** A public phaco video plays through the pipeline. Phase bar advances; pupil trace flat; risk band stays *routine*; no alerts. Point: "quiet on routine cases" (false alerts per case on screen).
3. **Complex case (1:00-2:20).** A public case with pupil constriction / radial folds:
   * During capsulorhexis the radial-fold indicator lights with the model's one-line explanation.
   * During phaco the pupil trace dips; the constriction indicator lights; band moves to *rising*.
   * Phaco timer crosses the 85th percentile; band moves to *high*.
   * Emphasise the rate limiting: indicators, not a siren.
4. **Case end (2:20-2:50).** Timeline JSON appears; the local LLM writes the complication-risk record; the verifier marks every statement as checked.
5. **Numbers (2:50-3:00).** One slide: external-test metrics; if MNGHA video was released, before/after adaptation on Saudi eyes.

## Fallback

`outputs/demo/phacoguard_demo_v1.mp4`: screen recording of steps 2-4 on public video, recorded before the event. Played if the venue machine, display, or environment fails. Never recorded inside the MNGHA environment unless the organisers permit it; never contains institutional data.

## Live demo checklist

- [ ] Weights cached locally; `python scripts/30_run_pipeline.py --video demo/routine.mp4 --offline` runs end to end with Wi-Fi off
- [ ] Second monitor or window layout tested at venue resolution
- [ ] Two public demo videos chosen (routine; complex) and pre-decoded
- [ ] Fallback video on the laptop and on a USB drive
- [ ] Metrics slide exported as PNG in case the deck fails
- [ ] All UI text is observational (no "do X" phrases)

## Recording the fallback video

```bash
python scripts/30_run_pipeline.py --video demo/complex.mp4 --offline --record outputs/demo/phacoguard_demo_v1.mp4
```

---

# Section 2a: annotation-driven mock demo

*Not the rule 3.3 working prototype. This is a demonstration in which every indicator fires at a moment
taken from an expert dataset label. It must never be presented as a model-driven prototype.*

Every frame and the end-of-case record carry the banner:

> Indicators driven by expert dataset labels; model outputs replace them as training completes.

## What drives each element

| Element | Driven by | Status |
|---|---|---|
| Phase bar | Cataract-1K `case_<id>_annotations_phases.csv`, frame by frame | **Label-driven** |
| Prolonged phaco | Accumulated labelled phaco time crossing the p85 / p95 of the labelled-case distribution | **Label-driven** |
| Risk band | Rule-based fusion over the above, 30 s per-marker rate limit | **Label-driven** |
| Pupil constriction | — | **No labelled source** (see below) |
| Radial folds | — | **No labelled source** (Tongren access pending) |

### Why pupil constriction has no labelled source

HANDOFF.md section 2a assumes the Cataract-1K "pupil reaction" subset carries annotated intervals.
It does not. As published on Synapse (`syn53395402`) the subset is **38 mp4 files and no annotation
file**: the label is at case level — *this case exhibits a pupil reaction* — with nothing saying when.
The 38 pupil-reaction cases also share **no case IDs** with the 56 phase-annotated cases, so those
videos carry no phase labels either.

Placing the indicator at a moment of our own choosing is exactly what section 2a forbids, so the
marker reports `no_labelled_source`. The dashboard renders that state in grey and distinct from
"not observed", so an absent data source can never be read as a negative finding.

## Running it

```bash
export SYNAPSE_AUTH_TOKEN=...   # personal access token, Download scope
python scripts/00_download_public_data.py --subset phase --accept-licence
python scripts/00_download_public_data.py --subset phase --accept-licence --videos case_5015 case_5353
python scripts/30_run_pipeline.py --mock --data-root data/cataract1k
```

The runner picks the median-phaco case (routine segment) and the longest-phaco case (complex segment)
unless `--cases` names others, writes `outputs/demo/timeline_<case>.json` per case, and renders
`outputs/demo/phacoguard_mock_v1.mp4`.

## Attribution

Cataract-1K is CC BY 4.0 and attribution must appear wherever the footage does. The dataset name,
case ID and licence are rendered on every frame and on each title card. Demo videos are not committed
to the repository (`.gitignore`), so redistribution stays a deliberate act.

## Result of the first run (20 Sep 2026)

Phaco-duration distribution over all **56 labelled cases**: p50 = 92 s, p85 = 165 s, p95 = 201 s.

| Segment | Case | Labelled phaco | Percentile | Events | Final state |
|---|---|---|---|---|---|
| Routine | `case_5015` | 93 s | 50 | none | routine |
| Complex | `case_4859` | 275 s | 98 | p85 at 04:47, p95 at 05:23 (case time) | rising |

`outputs/demo/phacoguard_mock_v1.mp4` — 99 s at 25 fps, 1920x1080, playing each case at 10x.

The band reaches *rising*, not *high*: `RiskFusion.state` requires a pupil or radial-fold event for
*high*, and neither has a labelled source yet. This is a faithful consequence of the available labels,
not a bug — the storyboard above reaches *high* only once a second marker has a source.

### Verified against the real annotation files

The published phase vocabulary is 12 names, confirmed across all 56 files:
`Incision, Viscoelastic, Capsulorhexis, Hydrodissection, Phacoemulsification, Irrigation/Aspiration,
Capsule Pulishing, Lens Implantation, Lens positioning, Viscoelastic_Suction,
Anterior_Chamber Flushing, Tonifying/Antibiotics`.

`Capsule Pulishing` is misspelled in the dataset; `configs/phase_map.yaml` matches the dataset's
spelling and `test_phase_map_covers_the_real_vocabulary` locks it. There is no `Idle` label — idle
time is simply unlabelled, so `phase_at()` returns `idle` for gaps between intervals.

The annotation CSVs carry `sec`/`endSec` columns alongside frame indices; the reader prefers the
dataset's own seconds and falls back to `frame / fps`.

### Colour semantics

Red is reserved for the `high` case risk state. A marker that has fired is rendered in the neutral
attention colour, because a red indicator beside a `routine` band reads as a contradiction: the
marker reports an observation, the band reports the aggregate state, and one firing of a single
marker does not leave `routine` under `RiskFusion.state`.

## Four segments (20 Sep 2026)

| # | Case | Driven by | Unavailable, and stated in the panel |
|---|---|---|---|
| 1 | `case_5015` | expert phase labels; routine, nothing fires | pupil, radial folds |
| 2 | `case_4859` | expert phase labels; prolonged phaco | pupil, radial folds |
| 3 | `case_742` | measured pupil-area decrease | phase bar, phaco timing, radial folds |
| 4 | `case_800` | measured pupil-area decrease | phase bar, phaco timing, radial folds |

202 s at 25 fps.

**Empty panels state their reason.** A blank or greyed panel reads as "nothing found"; these read
as "nothing to look with" — *phase bar unavailable: no phase labels for this case*, *pupil marker
unavailable: no pupil-reaction annotation for this case*.

**Every segment carries a source caption** naming dataset, case, licence and what drives each
indicator, and **the footer describes its own segment**. Segments 1-2 carry the label-driven banner;
segments 3-4 carry "Pupil marker is a measurement, not an expert annotation; SAM 2 confirmation
pending." One banner for both would have been false on whichever it did not describe.

### Episodes, not repeated alerts

A 30 s running maximum decays towards a sustained low, so a pupil that constricts and stays
constricted re-crosses the threshold repeatedly. `case_742` produced **seven threshold crossings for
two episodes**: one of 14 s, and one of **249 s reaching 69% below baseline** that merged six
crossings. Across those six the window maximum fell 0.45 to 0.34 while the area fell 0.24 to 0.17,
and only 11% of clean samples recovered to 85% of baseline. `case_800`: four crossings, three
episodes.

Two events merge when the gap between them is short **and** the pupil does not recover to 85% of the
episode baseline. Recovery is the deciding test: a pupil that recovers and constricts again is a new
episode, not a continuation.

