# PhacoGuard

**An AI second pair of eyes for complex cataract surgery.**
Research prototype for surgeon *awareness* during phacoemulsification. Entry for the BeyondVision MNGHA Ophthalmology Hackathon 2026.

PhacoGuard watches the operating-microscope video and flags three evidence-based candidate risk markers of a complex case:

1. **Pupil constriction** during phaco or cortex removal
2. **Anterior capsule radial folds** during capsulorhexis (early sign of zonular weakness)
3. **Prolonged phaco time** relative to the expected distribution (dense nucleus)

The markers combine into a calibrated **case risk band** (routine / rising / high) on a second monitor. At case end a locally hosted, open-weight language model writes a **complication-risk record** from the structured event timeline only; every number is verified against the timeline by code.

> **Classification (hackathon rule 5.3):** early-stage research prototype. Not a medical device. Never used for diagnosis, triage, surgical planning, or treatment decisions. Outputs are observations and a risk state, never instructions to the surgeon.

## Repository map

| Path | Contents |
|---|---|
| `docs/ARCHITECTURE.md` | Architecture summary (hackathon deliverable) |
| `docs/model_cards/` | AI/ML model metadata, one card per model (hackathon deliverable) |
| `docs/DEMO.md` | Live demo script and recorded-video fallback plan |
| `docs/AI_DISCLOSURE.md` | Generative AI transparency statement (rule 5.1.2) |
| `docs/DATA_AND_LICENSES.md` | Datasets, licences, split policy |
| `docs/COMPLIANCE.md` | Mapping of hackathon rules to design decisions |
| `configs/` | YAML configs for data, models, thresholds |
| `src/phacoguard/` | Package: data, models, detectors, pipeline, report |
| `scripts/` | Command-line entry points (extract, cache, train, eval, run) |
| `tests/` | Unit tests for phase mapping, detector logic, record verification |
| `CLAUDE.md` | Non-negotiable rules for any AI coding assistant working in this repo |

## Quick start (public data only)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python scripts/00_download_public_data.py --root /mnt/hdd/datasets      # public datasets only
python scripts/01_extract_frames.py --config configs/data.yaml
python scripts/02_cache_features.py --config configs/data.yaml
python scripts/10_train_segmentation.py --config configs/seg.yaml
python scripts/11_train_phase.py --config configs/phase.yaml
python scripts/12_train_detectors.py --config configs/detectors.yaml
python scripts/20_evaluate.py --split external                          # Cataract-101 held out
python scripts/30_run_pipeline.py --video path/to/public_case.mp4 --offline
```

## Two-environment rule

* **Environment A (team hardware or rented cloud GPU):** public datasets only.
* **Environment B (MNGHA sanctioned environment):** institutional data only; nothing leaves; the same code runs with `--offline`.

See `docs/COMPLIANCE.md`.

## Status

Pre-built pipeline declared at registration under rule 3.2. In-event work: adaptation and validation on MNGHA data, fusion calibration, live demonstration. Repository tag `pre-event-v1` marks the boundary.
