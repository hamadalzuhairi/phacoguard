# Model card: Prolonged phaco-time marker

*AI/ML model metadata, hackathon deliverable (rule 3.3). Metrics are filled by `scripts/21_update_model_cards.py` from `outputs/eval/`.*

| Field | Value |
|---|---|
| Model name | Prolonged phaco-time marker |
| Version | 0.1.0 (pre-event) |
| Architecture / method | Empirical duration distribution (no learned parameters); flags at 85th and 95th percentile; optional MNGHA prior |
| Training data | Phase timestamps from Cataract-1K, Cataract-LMM, Cataract-101 |
| Evaluation data | Cataract-101 timestamps |
| Metrics reported | Phaco-duration error (s) vs ground-truth timestamps; percentile agreement |
| Results | _pending: filled from outputs/eval/phaco_duration.json_ |
| Intended use | Research prototype for surgeon awareness within PhacoGuard; not for diagnosis, triage, surgical planning, or treatment decisions (hackathon rule 5.3) |
| Out-of-scope use | Any standalone clinical use; any use on patients without SFDA approval and institutional oversight |
| Data governance | Trained on public licensed data only; MNGHA data used only for adaptation/validation inside the sanctioned environment (rules 4.1, 4.2, 5.2) |
| Known limitations | Depends entirely on phase model; surgeon speed varies by site |
| Licence of weights | Same as base model family (see architecture); training code MIT (this repo) |
| Generative AI assistance | Training/evaluation code drafted with Claude; see docs/AI_DISCLOSURE.md |
