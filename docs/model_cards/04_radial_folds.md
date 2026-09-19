# Model card: Anterior capsule radial-fold marker (zonular weakness)

*AI/ML model metadata, hackathon deliverable (rule 3.3). Metrics are filled by `scripts/21_update_model_cards.py` from `outputs/eval/`.*

| Field | Value |
|---|---|
| Model name | Anterior capsule radial-fold marker (zonular weakness) |
| Version | 0.1.0 (pre-event) |
| Architecture / method | 3D-CNN (r2plus1d-18, torchvision, BSD) clip classifier; optional Qwen3-VL 4B LoRA (Apache-2.0) for one-line explanations |
| Training data | Tongren-Zonular-Video (537 CCC clips; licence to be confirmed on the Hugging Face card before training) |
| Evaluation data | Held-out Tongren clips; qualitative check on other datasets' capsulorhexis clips |
| Metrics reported | AUROC, F1, precision, recall, false alerts per case |
| Results | _pending: filled from outputs/eval/radial_folds.json_ |
| Intended use | Research prototype for surgeon awareness within PhacoGuard; not for diagnosis, triage, surgical planning, or treatment decisions (hackathon rule 5.3) |
| Out-of-scope use | Any standalone clinical use; any use on patients without SFDA approval and institutional oversight |
| Data governance | Trained on public licensed data only; MNGHA data used only for adaptation/validation inside the sanctioned environment (rules 4.1, 4.2, 5.2) |
| Known limitations | Trained on Chinese/Singaporean video; cross-population transfer untested; only fires during capsulorhexis phase |
| Licence of weights | Same as base model family (see architecture); training code MIT (this repo) |
| Generative AI assistance | Training/evaluation code drafted with Claude; see docs/AI_DISCLOSURE.md |
