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
