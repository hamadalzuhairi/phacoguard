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
