# Rules for AI coding assistants working in this repository

These rules are absolute. They come from the BeyondVision MNGHA Hackathon terms and from Saudi PDPL.

1. **Public data only on this machine.** Never download, copy, cache, or reference MNGHA/KAIMRC/KKESH institutional data here. If a path or file looks like patient data, stop and ask.
2. **No external AI or network calls in the pipeline.** All models are open-weight and run locally. `src/phacoguard` must run with the network disabled. Do not add API clients.
3. **Licences.** *Code and model weights:* permissive only — Apache-2.0, MIT, BSD. No AGPL dependencies (e.g. Ultralytics YOLOv8). *Datasets:* CC BY 4.0 preferred; **CC BY-NC 4.0 is permitted for non-commercial research use only** (team decision, 20 Sep 2026). **NoDerivatives (ND) datasets are not permitted** — this excludes Cataract-LMM (CC BY-NC-ND 4.0). A dataset with no published licence may not be used until terms are obtained in writing. Check `docs/DATA_AND_LICENSES.md` before adding any dataset or model.
4. **Language of outputs.** The system produces observations and a risk state for surgeon awareness. Never write UI text, comments, or docs that instruct the surgeon ("do X"), and never call it "guidance".
5. **Splits are by video, never by frame.** Cataract-101 is the external test set and must never be used for training or threshold tuning.
6. **Log AI assistance.** Any material code generated with an AI assistant is noted in `docs/AI_DISCLOSURE.md` (rule 5.1.2).
7. **Do not touch `docs/model_cards/*.md` metrics by hand.** They are filled from `outputs/eval/*.json` by `scripts/21_update_model_cards.py`.
