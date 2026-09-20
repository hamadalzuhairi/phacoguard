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

---

## Quick phase model (v0.1-quick, 20 Sep 2026)

**Not the model of record.** Its only job is to give the pupil-reaction videos a phase bar so the pupil marker can be gated. Everything it produces is labelled "phases predicted by a quick model (n=18 training cases, test accuracy 88%); full model pending".

| | |
|---|---|
| Encoder | ResNet-18 (torchvision, BSD), ImageNet weights, frozen |
| Sampling | 0.5 fps, 160 px, sequential OpenCV decode |
| Head | bidirectional GRU (128) + linear, 11 classes |
| Data | 24 shortest labelled cases, split **by video** 18/3/3 (rule 5) |
| Early stopping | on validation accuracy, patience 15 |
| Val accuracy | 87.7% |
| **Test frame accuracy** | **87.6%** (435 frames, 3 held-out videos) |

### Per-step F1

| step | F1 | support |
|---|---|---|
| incision | 0.762 | 11 |
| ovd_injection | 0.500 | 14 |
| capsulorhexis | 0.790 | 33 |
| hydrodissection | 0.722 | 22 |
| phaco | 0.985 | 98 |
| cortex_removal | 0.994 | 78 |
| capsule_polishing | 0.800 | 11 |
| iol_insertion | 0.809 | 25 |
| ovd_removal | 0.866 | 48 |
| wound_closure | 0.800 | 28 |
| transition | 0.819 | 67 |

The two steps the pupil gate depends on are the strongest: **phaco 98/98** and **cortex_removal 77/78**. That matters more than the headline accuracy, because the gate only consults those steps plus capsule_polishing.

`ovd_injection` is the single class below F1 0.60 (0.50, 5 of 14 correct), confusing mostly with capsulorhexis and incision — its temporal neighbours, and the shortest step in the workflow. It is merged into capsulorhexis **for the phase bar only**; all ten steps keep their own row in the step list, and the caption names the pair the model cannot yet distinguish.

ffmpeg was specified for decoding but is not used: it is LGPL, outside the permissive set rule 3 allows, and sequential OpenCV `grab()` measured 2.9 min for 7,311 s of video against 3.4 min for per-frame seeking, so it bought nothing.

Metrics here are copied from `outputs/phase/phase_report.json`, which `scripts/11_train_phase.py` writes; they are not hand-entered (rule 7).

