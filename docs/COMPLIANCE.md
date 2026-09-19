# Hackathon rules mapped to design decisions

| Rule | Requirement | PhacoGuard design |
|---|---|---|
| 3.1 / 3.2 | Original work; pre-existing projects declared and significantly extended in-event | Pipeline pre-built on open-source assets and declared at registration; tag `pre-event-v1`; in-event work = MNGHA adaptation, fusion calibration, validation, demo |
| 3.3 | Repo, architecture summary, model metadata, deck, demo | This repo; `docs/ARCHITECTURE.md`; `docs/model_cards/`; `deck/`; `docs/DEMO.md` |
| 4.1 / 4.2 | Institutional data only in sanctioned environment; no local copies | Two-environment design; `--offline` mode; `CLAUDE.md` rule 1 |
| 4.3 | Purge temporary files after the event | `scripts/99_purge_institutional.sh` |
| 5.1 | Permissive licences; disclose generative AI | `docs/DATA_AND_LICENSES.md`; `docs/AI_DISCLOSURE.md` |
| 5.2 | No clinical data to external AI | No network client in `src/`; local Qwen via Ollama/vLLM bound to localhost |
| 5.3 | Research prototype; no diagnosis/triage/planning | Observational UI text; risk band = awareness; README disclaimer |
| 6.2 / 6.3 | IP co-ownership triggers; clearance before patents/commercialisation | Algorithms and workflow built/validated on public data (Background IP); MNGHA data for validation only; any commercial step routed via MNGHA Innovation Division and KAIMRC IP Office |
| 7.1 | Research requires IRB | Roadmap: IRB protocol before any prospective study |
