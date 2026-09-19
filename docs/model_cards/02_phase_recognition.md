# Model card: Surgical phase recognition (7 phases)

*AI/ML model metadata, hackathon deliverable (rule 3.3). Metrics are filled by `scripts/21_update_model_cards.py` from `outputs/eval/`.*

| Field | Value |
|---|---|
| Model name | Surgical phase recognition (7 phases) |
| Version | 0.1.0 (pre-event) |
| Architecture / method | ResNet-50 (torchvision, BSD, ImageNet weights) frozen encoder + GRU over 20 s; median filter |
| Training data | Cataract-1K phase subset (56 videos) + Cataract-LMM phase layer, mapped to 7 phases via configs/phase_map.yaml |
| Evaluation data | Cataract-101 (external, never trained on) |
| Metrics reported | Frame accuracy, per-phase F1, confusion matrix, sequence-order violations per video |
| Results | _pending: filled from outputs/eval/phase_recognition.json_ |
| Intended use | Research prototype for surgeon awareness within PhacoGuard; not for diagnosis, triage, surgical planning, or treatment decisions (hackathon rule 5.3) |
| Out-of-scope use | Any standalone clinical use; any use on patients without SFDA approval and institutional oversight |
| Data governance | Trained on public licensed data only; MNGHA data used only for adaptation/validation inside the sanctioned environment (rules 4.1, 4.2, 5.2) |
| Known limitations | Hydrodissection vs cortex confusion; idle-phase ambiguity; different hospital protocols |
| Licence of weights | Same as base model family (see architecture); training code MIT (this repo) |
| Generative AI assistance | Training/evaluation code drafted with Claude; see docs/AI_DISCLOSURE.md |
