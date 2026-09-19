# PhacoGuard — handoff brief for Claude Code

Paste this as the first message in Claude Code, or save it as `HANDOFF.md` in the repository root and say "read HANDOFF.md and CLAUDE.md, then continue the plan."

## 1. What we are building

**PhacoGuard: an AI second pair of eyes for complex cataract surgery.** A research prototype that analyses phacoemulsification microscope video and flags three evidence-based candidate risk markers of a complex case, for surgeon *awareness*:

1. **Pupil constriction** during phaco or cortex removal (normalised pupil area falling from its running maximum)
2. **Anterior capsule radial folds** during capsulorhexis (the early visible sign of weak zonules; common in pseudoexfoliation, which is frequent in Saudi patients)
3. **Prolonged phaco time** relative to the expected duration distribution (the fingerprint of a dense nucleus)

The markers combine into a calibrated **case risk band** (routine / rising / high) shown on a second monitor, rate-limited so no indicator repeats within 30 seconds. At case end a locally hosted open-weight LLM writes a **complication-risk record** from the structured event timeline only, and a verifier checks every number in the record against the timeline.

Context: entry for the BeyondVision MNGHA Ophthalmology Hackathon (Riyadh). Application submitted 20 Sep 2026; acceptance 1 Oct; hackathon day 13 Oct 2026 (single day). Judged blind on innovation, clinical and eye-care impact, feasibility, team capability, presentation.

**Not in scope (decided):** integration into microscope eyepieces or heads-up displays, phaco-machine telemetry, pre-operative risk scoring, PCR detection itself. Do not build or discuss these.

## 2a. FIRST TASK: annotation-driven mock demo (1–3 days, no training)

Before any model is trained, build a demo video in which the dashboard runs over real public surgical footage and the indicators fire at moments defined by **expert dataset labels, never by the team's or Claude's own judgement** (no clinician on the team):

- Pupil constriction → Cataract-1K "pupil reaction" irregularity subset: use the annotated intervals as-is.
- Radial folds → Tongren-Zonular-Video: use a clip labelled positive by the dataset's ophthalmologists (clips are ~7 s).
- Prolonged phaco → phase labels (Cataract-1K / Cataract-101 / Cataract-LMM): compute the phaco-duration distribution and pick a case above the 85th percentile.
- Phase bar → the same phase labels, frame by frame.

Expect to chain two or three labelled cases, each named on screen with its dataset and case ID, rather than one surgery. Claude may inspect frames to confirm the overlay aligns with the label, but the label is ground truth. Do not use YouTube or teaching videos (no licence, no structured labels).

Label it everywhere as **"indicators driven by expert dataset labels; model outputs replace them as training completes."** It is not the rule 3.3 working prototype and must never be presented as one. Reuse the same footage and layout for the real, model-driven recording later.

## 2. The immediate goal

**A demo video, made entirely from open-source surgical videos, showing a complex cataract case playing through PhacoGuard while the software flags the markers for the surgeon.** Concretely:

- Input: a public phaco video with a visible complication or risk sign (small pupil, pupil constriction, radial folds, long phaco). Candidates: Cataract-1K "pupil reaction" irregularity subset; capsulorhexis clips from Tongren-Zonular-Video; long-phaco cases from Cataract-101 / Cataract-LMM.
- Output: a screen recording (`outputs/demo/phacoguard_demo_v1.mp4`) of the dashboard: phase bar advancing, pupil-area trace, three indicators lighting with timestamps and a one-line reason, risk band moving routine → rising → high, then the end-of-case record appearing with the verifier's checks. Include a second short segment on a routine case where nothing fires (false alerts per case on screen).
- Everything runs offline on our own hardware. This video is the rule 3.3 "recorded prototype demonstration" fallback and the centrepiece of the pitch.

Build order is therefore: data → segmentation → phase → detectors → fusion → dashboard → record writer → recording. Validation numbers come from held-out public data (Cataract-101 is the external test set, never trained on).

## 3. Non-negotiable rules (also in CLAUDE.md)

1. Public datasets only on our machines. Never touch MNGHA/KAIMRC/KKESH data here; that happens only inside their sanctioned environment on hackathon day, with the same code in `--offline` mode.
2. No network calls anywhere in `src/phacoguard`. All models open-weight and local (Qwen3 via Ollama/vLLM bound to localhost).
3. Permissive licences only: Apache-2.0, MIT, BSD, CC BY 4.0. No Ultralytics YOLOv8 (AGPL). Use PIDNet / YOLOX / torchvision models.
4. Language: outputs are observations and a risk state. Never write UI text, comments, or docs that instruct the surgeon; never call it "guidance" (hackathon rule 5.3).
5. Splits by video, never by frame. Cataract-101 is external test only.
6. Log any material AI-assisted code in `docs/AI_DISCLOSURE.md` (rule 5.1.2).
7. Do not edit metrics in `docs/model_cards/*.md` by hand; they are filled from `outputs/eval/*.json`.

## 4. Repository

`https://github.com/hamadalzuhairi/phacoguard` (skeleton already generated; if empty, unzip `phacoguard_repo.zip` and push). Key files:

- `README.md`, `CLAUDE.md`, `docs/ARCHITECTURE.md`, `docs/DEMO.md`, `docs/DATA_AND_LICENSES.md`, `docs/COMPLIANCE.md`, `docs/AI_DISCLOSURE.md`, `docs/model_cards/*.md`
- `configs/data.yaml`, `configs/phase_map.yaml` (7 phases: incision, capsulorhexis, hydrodissection, phaco, cortex_removal, iol_insertion, idle), `configs/seg.yaml`, `configs/phase.yaml`, `configs/detectors.yaml`, `configs/timeline_schema.json`
- `src/phacoguard/`: `data/phase_map.py`, `data/splits.py`, `models/phase_gru.py`, `detectors/pupil.py`, `detectors/phaco_time.py`, `pipeline/fusion.py`, `report/verify.py` (fusion and verifier have passing tests in `tests/`)
- `scripts/00_download_public_data.py` … `30_run_pipeline.py` (stubs to implement in order), `scripts/99_purge_institutional.sh`

## 5. Data

| Dataset | Use | Licence |
|---|---|---|
| Cataract-1K (Sci Data 2024; 1,000 videos; phase labels for 56; masks for 2,256 frames; pupil-reaction + IOL-rotation subsets) | segmentation, phase, pupil-constriction positives, demo footage | CC BY 4.0 |
| CaDIS (4,670 masked frames from CATARACTS) | segmentation | challenge terms — confirm |
| Cataract-101 (101 videos, phase labels) | external test set only | research terms — confirm |
| Cataract-LMM (~3,000 videos; phase/segmentation layers; Hugging Face `mjahmadi/Cataract-LMM`) | phase, segmentation | check card |
| Tongren-Zonular-Video (Hugging Face `leonChen/Tongren-Zonular-Video`; 537 capsulorhexis clips labelled for radial folds) | radial-fold detector, demo footage | **check card before training; the preprint is CC BY-ND** |

Raw videos live on the 5 TB external HDD; extracted frames (5 fps, 512×324) and cached features on the NVMe. Budget: tens to hundreds of GB.

## 6. Hardware and environment

- PC: Ryzen 7 7700, 32 GB DDR5, **AMD Radeon RX 7800 XT 16 GB (ROCm, not CUDA)**, 1 TB NVMe, 5 TB external HDD, Windows 11.
- Plan: test Windows-native PyTorch with ROCm for one day; if unreliable, dual-boot Ubuntu 24.04 with the official ROCm PyTorch wheels. Do not use the `HSA_OVERRIDE_GFX_VERSION` hack.
- Consequences: use bf16 (no bitsandbytes 4-bit training); SDPA instead of Flash-Attention; keep models small (Qwen3 4B for the record writer; Qwen3-VL 2B/4B if explanations are added). Free-tier Kaggle GPU can run the radial-fold job in parallel.
- No paid cloud. Long jobs run overnight in `tmux` with per-epoch checkpoints. One training job at a time on the GPU; the LLM server is off during training.
- Remote access for the teammate: Tailscale + SSH; Ollama bound to the Tailscale IP only.

## 7. Models

| Component | Choice |
|---|---|
| Segmentation (pupil, iris, instruments) | PIDNet-S fine-tuned on CaDIS + Cataract-1K masks; fallback torchvision DeepLabV3-ResNet50 |
| Phase recognition | frozen torchvision ResNet-50 features cached once → GRU over 20 s window; median filter 3 s |
| Pupil constriction | windowed features on normalised pupil area (drop from running max, slope, variance) → gradient-boosted classifier; threshold learned on Cataract-1K pupil-reaction subset; gated to phaco/cortex phases |
| Radial folds | r2plus1d-18 clip classifier on 3 s clips, gated to capsulorhexis phase; optional Qwen3-VL LoRA for one-line explanations |
| Prolonged phaco | empirical percentiles (85th, 95th) of phaco-phase duration from public videos |
| Fusion | logistic fusion + isotonic calibration; states routine/rising/high; 30 s rate limit per detector |
| Record writer | Qwen3 4B, LoRA on synthetic timeline→record pairs, fixed template; served via Ollama on localhost |
| Verifier | regex/field matching of every number in the record against the timeline JSON |

Targets (from published benchmarks): phase accuracy ~90% on Cataract-101; pupil Dice 88–90%; radial-fold AUROC ~0.9; ≥15 fps end to end on the laptop GPU.

## 8. Evaluation to report

Pupil Dice/IoU; phase accuracy, per-phase F1, confusion matrix, phase-order violations per video; per-detector precision, recall, F1, AUROC, lead time (s), false alerts per case; phaco-duration error (s); record factual accuracy (% verified); latency per frame and fps. Thresholds tuned on validation only; test and external reported once.

## 9. Order of work (26 days, PC only)

1. Environment: ROCm test, Tailscale, Ollama, dataset downloads to HDD (check licences first).
2. Frame extraction + phase-map table + video-level splits; cache ResNet-50 features (overnight).
3. Segmentation training (overnight); phase GRU on cached features (minutes per run).
4. Three detectors; fusion + calibration; metrics on held-out and external sets.
5. Dashboard (FastAPI + static front end): phase bar, pupil trace, three indicators, risk band, event log, record panel. All UI text observational.
6. Record writer LoRA + verifier.
7. `scripts/30_run_pipeline.py --video … --offline --record` → the demo video (routine case, then complex case).
8. Package for offline use (cached weights, pinned deps), fill model cards from eval JSON, update AI disclosure, tag `pre-event-v1`.

## 10. Open items

- Confirm licences for Tongren-Zonular-Video, CaDIS, Cataract-101, Cataract-LMM before the first training run.
- Confirm which public videos best show pupil constriction and radial folds; shortlist 3–5 for the demo.
- Ask the organisers (already in progress): event length, GPU in the sanctioned environment, whether surgical video is in the MNGHA release, whether fine-tuned weights may leave the environment.
