# Model card: Pupil-constriction marker

*AI/ML model metadata, hackathon deliverable (rule 3.3). Metrics are filled by `scripts/21_update_model_cards.py` from `outputs/eval/`.*

| Field | Value |
|---|---|
| Model name | Pupil-constriction marker |
| Version | 0.1.0 (pre-event) |
| Architecture / method | Gradient-boosted classifier (scikit-learn/XGBoost, BSD/Apache-2.0) on windowed normalised pupil-area features |
| Training data | Cataract-1K pupil-reaction irregularity subset (positives) + routine videos (negatives) |
| Evaluation data | Held-out Cataract-1K clips |
| Metrics reported | Precision, recall, F1, AUROC, lead time (s), false alerts per case |
| Results | _pending: filled from outputs/eval/pupil_constriction.json_ |
| Intended use | Research prototype for surgeon awareness within PhacoGuard; not for diagnosis, triage, surgical planning, or treatment decisions (hackathon rule 5.3) |
| Out-of-scope use | Any standalone clinical use; any use on patients without SFDA approval and institutional oversight |
| Data governance | Trained on public licensed data only; MNGHA data used only for adaptation/validation inside the sanctioned environment (rules 4.1, 4.2, 5.2) |
| Known limitations | Labels are event-level, onset timestamps approximate; segmentation errors propagate |
| Licence of weights | Same as base model family (see architecture); training code MIT (this repo) |
| Generative AI assistance | Training/evaluation code drafted with Claude; see docs/AI_DISCLOSURE.md |

---

## Measured pupil-area decrease (v0.2.0, 20 Sep 2026)

The marker no longer claims an expert annotation. Cataract-1K's pupil-reaction subset carries a
**case-level** clinician flag and no interval labels (verified: the subset is 38 `.mp4` files, and the
whole 5,823-file project contains annotation files only under `Phase_recognition_dataset` and
`Segmentation_dataset`). The marker is therefore a **measurement**, described everywhere as
"measured pupil-area decrease, ungated (no phase labels for this case)".

It is **ungated**: the 38 pupil-reaction cases share no case IDs with the 56 phase-annotated cases, so
the phaco/cortex gate cannot be applied. `drop_events` records `phase=None`, and the gate stays
available for a re-run once the phase model can supply predicted phases.

### Method

| Stage | Choice |
|---|---|
| Screener | Classical, `src/phacoguard/detectors/pupil_classical.py` — Otsu on the LAB **a\*** (red-green) channel inside a Hough limbus circle fixed from frame 0; largest component; fitted ellipse area |
| Confirmation | SAM 2 zero-shot video tracking (Apache-2.0), `scripts/12_track_pupil_sam2.py`, one point prompt on the pupil in frame 0 |
| Normalisation | Pupil area over the Hough limbus disc area. The detector measures a *relative* fall, so a constant reference cancels |
| Occlusion filter | circularity ≥ 0.55, ellipse fill ratio ≥ 0.80, specular fraction ≤ 0.12, pupil/limbus ≤ 0.60 |
| Drop rule | > 20% below the 30 s running maximum, sustained ≥ 10 s over clean frames |
| Density rule | ≥ 60% window coverage by clean samples, and ≥ 5 clean samples on each side of the candidate |

### Two segmentation choices, and the artefacts that forced them

**1. Red reflex, not darkness.** The first implementation thresholded the darkest 10-15% of pixels
inside the limbus, on the assumption that the pupil is dark. Under coaxial microscope illumination it
is the opposite: the pupil is the bright red-reflex disc, and the darkest pixels inside the limbus are
the limbus rim and the instruments. That threshold returned a *ring*, median circularity 0.22, and
**0 of 974 frames passed the occlusion filter**. Otsu on the a\* channel replaced it.

**2. Otsu, not a fixed percentile.** Any fixed percentile ("reddest 25%") pins the measured area to a
constant fraction of the limbus and suppresses exactly the changes the detector exists to find. Otsu
is data-driven, so the area is free to move.

### Two rejected artefacts

**Start-of-video regime change.** In `case_709` the measured area ran 0.27 → 0.75 → 0.27 within the
first 42 s, producing a 71% "drop" that ranked first. A pupil cannot triple and then halve in 20 s;
the threshold had escaped the pupil while the view was still unstable. Rejected by the anatomical
plausibility gate: a maximally dilated ~8 mm pupil against a ~12 mm limbus gives an area ratio near
0.45, so readings above 0.60 are not a pupil. `case_712` showed the same artefact at t = 5.8 s (65%).

**Sparse-window inflation.** The occlusion filter removes roughly half of all frames and the survivors
arrive in clumps, so a 30 s running maximum can rest on a handful of unevenly spaced samples. In
`case_709` this produced an apparent 48% step-down at t = 438 s that disappears once the window is
required to be well covered. The density rule above exists for this.

### Status

Not yet validated. The SAM 2 confirmation run and the correlation against the classical screener are
outstanding, and no ranking across the 38 videos has been produced. Any shortlist drawn from fewer
than all 38 must be labelled "clearest among N measured", never "clearest".

Metrics remain the property of `scripts/21_update_model_cards.py` (rule 7); nothing above is a metric.

### Screening pass, 20 Sep 2026

**ranked by classical screener across 31 of 38 (7 not retrieved); SAM 2 confirmation pending.**

31 of the 38 pupil-reaction videos were screened at 2 fps, 33 s per video. Seven could not be
retrieved from Synapse: the ZIP packager caps package size and the multi-file fallback did not start.
Missing: `case_8157, case_8171, case_8228, case_8297, case_8316, case_8347, case_8349`.

Shortlist for SAM 2 confirmation (`scripts/14_sam2_confirm.py`, workstation only):

| # | Case | Drop | Clean frames | First event |
|---|---|---|---|---|
| 1 | case_769 | 69.3% | 33% | 252 s |
| 2 | case_712 | 66.6% | 44% | 110 s |
| 3 | case_8167 | 65.6% | 25% | 288 s |
| 4 | case_730 | 38.9% | 29% | 560 s |
| 5 | case_709 | 36.2% | 44% | 349 s |

All five are **provisional**. `case_709` at 349 s / 36.2% and `case_712` at 110 s / 66.6% remain
flagged provisional until SAM 2 confirms them.

**Only 5 of 31 cases produced any event.** That is not a clean negative for the other 26. The
occlusion filter rejects roughly three quarters of frames (median clean fraction ~25%, and seven
cases fall below 10%, two at 0%). Below about 20% clean, a zero drop means the data could not support
a judgement, not that no constriction occurred. **Absence of an event here is not evidence of
absence**, and the low clean fraction is itself the most important open problem for this marker: a
clinician-flagged pupil-reaction cohort should not be yielding 26 silent cases.

The gap between vetted and unvetted drops shows how much the rules carry: unvetted deepest drops run
to 97.9%, and 26 cases with an unvetted drop above 40% produce no vetted event at all.

### Adaptive segmentation and limbus tracking (20 Sep 2026)

Two changes, after the rejection diagnosis showed circularity at 63% of rejections was a *symptom*
of segmenting the wrong structure rather than a threshold being wrong.

**Adaptive mode.** Red-vs-dark is decided per 20 s window from the a\* contrast between the central
disc and the surrounding annulus, with hysteresis (enter red above +6, dark below +2, otherwise hold).
A single threshold at +4 sat in the middle of the contrast distribution and the mode flapped —
case_709 switched six times on contrasts of 1 to 19. Switches are recorded per video in
`screen_summary.json` as `{t_s, from, to, a_contrast}`: in a dense cataract the reflex is absent until
the nucleus is gone, so a dark→red switch marks that moment. 121 switches across 28 of 31 cases.

**Limbus tracking.** Re-estimated every 3 s, median-smoothed over the last 5 accepted estimates,
rejecting any candidate whose radius or centre moves more than 20%.

**Segmentation ROI.** Otsu now runs inside 0.80 of the limbus radius. The full disc includes the
white sclera ring, which dominated the histogram: in dark mode Otsu split sclera from everything
else and returned the whole iris plus pupil (measured ratio 0.86-0.99).

**Otsu comparison.** OpenCV defines the foreground as `src > t`, but the code used `>=`. When the
threshold landed on the iris value the iris was taken in with the pupil. Found by a unit test on a
synthetic eye, not by inspection.

**The specular test was removed.** Over 30,103 frames it rejected 3 while circularity rejected
14,587. `specular_fraction` is still recorded.

#### Result

| | Before | After |
|---|---|---|
| Clean frames | 23.2% | **50.8%** |
| Cases below 10% clean | 7 (two at 0%) | **0** |
| Events | 5 | **47** |
| Cases with an event | 5/31 | **18/31** |
| Plausibility rejections | 27.4% of frames | **3.9%** |
| Circularity rejections | 48.5% of frames | 43.0% (now 87.4% of all rejections) |

The limbus tracker mostly *declines* to update — case_854 accepted 1 of 320 candidates, case_848 3 of
250 — so it is acting as a stabiliser rather than a tracker, and most of the gain came from the ROI
restriction and the Otsu comparison fix. Hough remains an unreliable limbus estimator; that is the
next thing to replace, not to retune.

#### Circularity A/B, after both fixes

| Threshold | Clean | Events | Cases with an event |
|---|---|---|---|
| ≥ 0.55 (current) | 50.8% | 47 | 18/31 |
| ≥ 0.45 | 61.7% | 81 | 24/31 |
| ≥ 0.35 | 70.1% | 104 | 26/31 |

Not yet changed. Circularity is still 87% of rejections, so the threshold now genuinely matters — but
with no ground truth, a looser bound buys events of unknown quality. This is the decision SAM 2
confirmation exists to inform, and it should be made after that, not before.

#### Correction, 20 Sep 2026

An earlier version of this section carried a per-video ranking led by `case_848` at 97.5%. That was
read from a stale `screen_summary.json`: two full screening runs overlapped, and the summary was
sampled after the first (pre-Otsu-fix) run had written it but before the second finished. The
aggregate figures were unaffected, but the ranking was. The corrected leaders are `case_716` (89.0%),
`case_8197` (87.5%) and `case_8175` (80.3%); `case_848` is 70.2%. Screening runs must not overlap:
they share output paths.

