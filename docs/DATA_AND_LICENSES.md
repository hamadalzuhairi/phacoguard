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
