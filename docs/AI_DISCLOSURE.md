# Generative AI transparency statement (hackathon rule 5.1.2)

The team used Anthropic's Claude (claude.ai and Claude Code) for:

1. Generating and debugging training, evaluation, and pipeline code in this repository.
2. Executing and monitoring model training runs on the team's own hardware and free/rented cloud GPUs, using **public datasets only**.
3. Generating synthetic timeline-to-record pairs used to fine-tune the operative-record writer (no patient data involved).
4. Drafting and editing project documentation, the application text, and slides.

All system design, dataset and model selection, evaluation design, clinical framing, and final decisions were made by the team. **No institutional data was ever provided to Claude or any external AI service.** All processing of MNGHA data is performed inside the sanctioned environment using locally hosted open-weight models.

Log of material AI-assisted contributions (append as work proceeds):

| Date | Component | Assistance |
|---|---|---|
| 2026-09-19 | Repository skeleton, docs, configs | Drafted with Claude |
| 2026-09-20 | Licence verification of the five candidate datasets | Primary-source check with Claude; findings recorded in `docs/DATA_AND_LICENSES.md` |
| 2026-09-20 | `src/phacoguard/mock/` (label readers, event sources, timeline assembly) | Written with Claude Code |
| 2026-09-20 | `src/phacoguard/render/` (dashboard compositor, demo recorder) | Written with Claude Code |
| 2026-09-20 | `scripts/00_download_public_data.py`, `scripts/30_run_pipeline.py --mock` | Written with Claude Code |
| 2026-09-20 | `tests/test_mock_demo.py` (21 tests, no dataset required) | Written with Claude Code |
| 2026-09-20 | `scripts/12_track_pupil_sam2.py` (SAM 2 zero-shot pupil tracking) | Written with Claude Code |
| 2026-09-20 | `src/phacoguard/detectors/pupil_measured.py` + `tests/test_pupil_measured.py` | Written with Claude Code |
| 2026-09-20 | `src/phacoguard/detectors/pupil_classical.py` (Otsu a* screener) | Written with Claude Code |
| 2026-09-20 | `src/phacoguard/detectors/pupil_agreement.py` + `scripts/13`, `scripts/14` | Written with Claude Code |
| 2026-09-20 | Adaptive red/dark segmentation, limbus tracking, `scripts/15_diagnose_rejections.py` | Written with Claude Code |
