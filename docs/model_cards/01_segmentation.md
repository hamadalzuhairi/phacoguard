# Model card: Pupil and instrument segmentation

*AI/ML model metadata, hackathon deliverable (rule 3.3). Metrics are filled by `scripts/21_update_model_cards.py` from `outputs/eval/`.*

| Field | Value |
|---|---|
| Model name | Pupil and instrument segmentation |
| Version | 0.1.0 (pre-event) |
| Architecture / method | PIDNet-S (MIT) fine-tuned; fallback torchvision DeepLabV3-ResNet50 (BSD) |
| Training data | CaDIS (4,670 frames, 25 videos) + Cataract-1K masks (2,256 frames, 30 videos) |
| Evaluation data | Held-out CaDIS videos; Cataract-1K test videos |
| Metrics reported | Dice, IoU per class; pupil-area error vs ground truth |
| Results | _pending: filled from outputs/eval/segmentation.json_ |
| Intended use | Research prototype for surgeon awareness within PhacoGuard; not for diagnosis, triage, surgical planning, or treatment decisions (hackathon rule 5.3) |
| Out-of-scope use | Any standalone clinical use; any use on patients without SFDA approval and institutional oversight |
| Data governance | Trained on public licensed data only; MNGHA data used only for adaptation/validation inside the sanctioned environment (rules 4.1, 4.2, 5.2) |
| Known limitations | Glare, blood, transparent tissue; zoom changes (mitigated by limbus normalisation) |
| Licence of weights | Same as base model family (see architecture); training code MIT (this repo) |
| Generative AI assistance | Training/evaluation code drafted with Claude; see docs/AI_DISCLOSURE.md |
