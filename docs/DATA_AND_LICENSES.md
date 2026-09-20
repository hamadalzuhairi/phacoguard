# Datasets, licences, and split policy

| Dataset | Content | Licence / terms | Use in PhacoGuard |
|---|---|---|---|
| Cataract-1K (Ghamsarian et al., Sci Data 2024) | 1,000 videos; phase labels (56 videos); masks (2,256 frames); pupil-reaction and IOL-rotation subsets | CC BY 4.0 | Segmentation, phase, pupil-constriction marker |
| CaDIS (Grammatikopoulou et al., Med Image Anal 2021) | 4,670 masked frames from CATARACTS videos | CATARACTS/EndoVis challenge terms; **confirm before redistribution** | Segmentation |
| Cataract-101 (Schoeffmann et al., MMSys 2018) | 101 videos, phase labels | Research terms on dataset page; **confirm** | External test set only |
| Cataract-LMM (Ahmadi et al., Sci Data 2026) | ~3,000 videos; phase, segmentation, tracking, skill layers | Check Hugging Face card | Phase, segmentation |
| Tongren-Zonular-Video (Hugging Face) | 537 CCC clips labelled for radial folds | **Check card; associated preprint is CC BY-ND 4.0** | Radial-fold marker |
| MNGHA institutional data | as released on hackathon day | Rules 4.1-4.3 | Adaptation and validation only, inside sanctioned environment |

## Models

| Model | Licence | Role |
|---|---|---|
| PIDNet | MIT | segmentation |
| YOLOX | Apache-2.0 | alternative detector/segmenter |
| torchvision ResNet-50, DeepLabV3, r2plus1d | BSD-3 | encoders / fallbacks |
| SAM 2 | Apache-2.0 | mask propagation for extra training masks |
| Qwen3 / Qwen3-VL | Apache-2.0 | record writer; optional explanation model |
| Ultralytics YOLOv8 | AGPL-3.0 | **not used** (copyleft) |

## Split policy

* Splits are by video (and by surgeon when available), never by frame.
* Cataract-101 is fully external: never used for training, threshold selection, or calibration.
* Validation split is used for thresholds and early stopping; test split reported once.

## Licence verification log

Checked 20 Sep 2026 against primary sources (dataset pages / Hugging Face cards). No data was downloaded.

| Dataset | Licence as published | Access | Commercial | Derivatives | Meets CLAUDE.md rule 3? |
|---|---|---|---|---|---|
| Cataract-1K | CC BY 4.0 | Synapse account; accept conditions | Yes | Yes | **Yes** |
| Cataract-101 | CC BY-NC 4.0 | Open FTP, no registration | No | Yes | **No** (NC) |
| Tongren-Zonular-Video | CC BY-NC 4.0 | HF, gated `manual` (owner approval) | No | Yes | **No** (NC) |
| Cataract-LMM | CC BY-NC-ND 4.0 | HF, gated `auto` | No | **No** | **No** (NC + ND) |
| CaDIS | **Unresolved** — no licence published on the dataset pages | via CATARACTS on IEEE DataPort (free account) | Unknown | Unknown | **Unknown** |

Sources: Synapse syn53404507 and github.com/Negin-Ghamsarian/Cataract-1K; ftp.itec.aau.at/datasets/ovid/cat-101/;
huggingface.co/datasets/leonChen/Tongren-Zonular-Video (`gated: manual`, 223 mp4, ~521 GB);
huggingface.co/datasets/mjahmadi/Cataract-LMM (`cc-by-nc-nd-4.0`, `gated: auto`);
cataracts-semantic-segmentation2020.grand-challenge.org and ieee-dataport.org/open-access/cataracts.

### Open questions raised by this check

1. **Rule 3 conflict.** CLAUDE.md rule 3 permits Apache-2.0, MIT, BSD and CC BY 4.0 only. On a strict reading only
   Cataract-1K qualifies; the external test set (Cataract-101) and the radial-fold source (Tongren) are both NC.
   A team decision is needed on whether NC is acceptable for a non-commercial research prototype.
2. **Cataract-LMM is ND.** NoDerivatives is a stronger bar than NC: a fine-tuned model and an overlaid demo video are
   plausibly derivative works. Treat as unusable until clarified with the authors.
3. **CaDIS has no published licence.** The labels are derived from CATARACTS (IEEE DataPort, "open access", account
   required). Terms must be obtained from the organisers in writing before use.
4. **Tongren clip count.** HANDOFF.md records 537 clips; the card lists 223 mp4 files at ~521 GB. Confirm on access.
5. Tongren access is gated on manual owner approval — request early; it is on the critical path for radial folds.

## Tongren-Zonular-Video: repository contents (checked 20 Sep 2026)

Access request submitted 20 Sep 2026 (gate is `manual`; awaiting the authors' review). The public file
tree was read while the request was pending:

| | |
|---|---|
| Files in repo | **203** — 201 `.mp4`, plus `.gitattributes` and `README.md` |
| Total size | 521.2 GB |
| Video size | min 862 MB, median 2.2 GB, max 8.5 GB |
| Naming | numeric, `10.mp4`, `100.mp4`, `103.mp4`, … |
| **Label files** | **none** |

**There is no annotation file of any kind in the repository.** The videos are full-length surgeries,
not the ~7 s clips HANDOFF.md describes, and nothing in the repo records which are positive for
anterior capsular radial folds.

This is the same obstruction as the Cataract-1K pupil-reaction subset: the footage is published, the
expert labels are not. Both markers are blocked on the same thing — a label file that exists in the
authors' possession but not in the public release.

Next step for both: write to the authors.
- Radial folds — `cherishleon01@gmail.com` (listed on the dataset card for usage questions).
- Pupil constriction — the Cataract-1K authors (Ghamsarian et al.).

Ask for the clip/interval labels and their format, not for more video.

## Measured pupil-area decrease (SAM 2) — status 20 Sep 2026

The pupil marker is no longer label-driven. It is a **measurement** produced by SAM 2 zero-shot
tracking (Apache-2.0, permitted by rule 3) and must be described in the UI and the deck as
"measured pupil-area decrease", never as a clinician annotation. Its 20% / 30 s constants are the
team's choice.

Findings from the first run:

* **Pupil segmentation works well.** A single Hough-located centre click on frame 0 produces a clean
  pupil mask that SAM 2 propagates.
* **The limbus cannot be prompted the same way.** A second point on the iris makes SAM 2 return 92%
  of the frame. The limbus disc area from the Hough circle on frame 0 is used as a fixed normalising
  reference instead. Because the detector measures a *relative* fall from a running maximum, a
  constant reference cancels out.
* **Cost on this machine:** 3.27 s/frame (CPU, `sam2.1_hiera_tiny`, one tracked object). Roughly
  25-50 min per video at 1 fps. Tracking two objects cost 10.5 s/frame, so object count dominates.
  Per-frame cost rises with sequence length, so long videos are worse than this figure suggests.
* **Phase gating is impossible on these videos.** The 38 pupil-reaction cases carry no phase labels
  (no case-ID overlap with the 56 phase-annotated cases), so "during phaco or cortex" cannot be
  applied to them. `drop_events` records `phase=None` when ungated, and an ungated measurement is a
  weaker claim that must not be presented as a phase-gated one.
