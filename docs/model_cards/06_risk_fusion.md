# Model card: Risk fusion and calibration

*AI/ML model metadata, hackathon deliverable (rule 3.3). Metrics are filled by `scripts/21_update_model_cards.py` from `outputs/eval/`.*

| Field | Value |
|---|---|
| Model name | Risk fusion and calibration |
| Version | 0.1.0 (pre-event) |
| Architecture / method | Logistic fusion of detector confidences (+ optional pre-op prior) with isotonic calibration; 30 s per-detector rate limit; states routine/rising/high |
| Training data | Validation-split detector outputs; MNGHA paired cases if released (in-event) |
| Evaluation data | Test split; reliability plot |
| Metrics reported | Calibration (ECE), false alerts per case, state-transition counts |
| Results | _pending: filled from outputs/eval/risk_fusion.json_ |
| Intended use | Research prototype for surgeon awareness within PhacoGuard; not for diagnosis, triage, surgical planning, or treatment decisions (hackathon rule 5.3) |
| Out-of-scope use | Any standalone clinical use; any use on patients without SFDA approval and institutional oversight |
| Data governance | Trained on public licensed data only; MNGHA data used only for adaptation/validation inside the sanctioned environment (rules 4.1, 4.2, 5.2) |
| Known limitations | Fusion benefit over single markers unproven until paired data exists |
| Licence of weights | Same as base model family (see architecture); training code MIT (this repo) |
| Generative AI assistance | Training/evaluation code drafted with Claude; see docs/AI_DISCLOSURE.md |
