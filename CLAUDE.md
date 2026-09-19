# Rules for AI coding assistants working in this repository

These rules are absolute. They come from the BeyondVision MNGHA Hackathon terms and from Saudi PDPL.

1. **Public data only on this machine.** Never download, copy, cache, or reference MNGHA/KAIMRC/KKESH institutional data here. If a path or file looks like patient data, stop and ask.
2. **No external AI or network calls in the pipeline.** All models are open-weight and run locally. `src/phacoguard` must run with the network disabled. Do not add API clients.
3. **Permissive licences only.** Apache-2.0, MIT, BSD, CC BY 4.0. No AGPL dependencies (e.g. Ultralytics YOLOv8). Check `docs/DATA_AND_LICENSES.md` before adding any dataset or model.
4. **Language of outputs.** The system produces observations and a risk state for surgeon awareness. Never write UI text, comments, or docs that instruct the surgeon ("do X"), and never call it "guidance".
5. **Splits are by video, never by frame.** Cataract-101 is the external test set and must never be used for training or threshold tuning.
6. **Log AI assistance.** Any material code generated with an AI assistant is noted in `docs/AI_DISCLOSURE.md` (rule 5.1.2).
7. **Do not touch `docs/model_cards/*.md` metrics by hand.** They are filled from `outputs/eval/*.json` by `scripts/21_update_model_cards.py`.
