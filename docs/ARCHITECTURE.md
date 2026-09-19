# PhacoGuard: Architecture Summary

*Hackathon deliverable (rule 3.3): architecture summary. Companion: `docs/model_cards/` for per-model metadata.*

## 1. Purpose and scope

PhacoGuard is a research prototype that analyses phacoemulsification microscope video and produces (a) a live, calibrated case-risk band for surgeon awareness and (b) an end-of-case complication-risk record. It is not a medical device and issues no instructions (hackathon rule 5.3).

## 2. System overview

```
 microscope video (file or capture card, 15-25 fps)
        |
        v
 [1] Frame grabber  --->  [2] Segmentation (pupil, instruments)  --->  pupil area series
        |                        |
        |                        v
        +------------------> [3] Phase recognition (7 phases, 20 s context)
                                 |
        +------------------------+------------------------+
        v                        v                        v
 [4a] Pupil-constriction   [4b] Radial-fold clip     [4c] Phaco-duration
      detector                  classifier (CCC only)     percentile rule
        \                        |                        /
         \                       v                       /
          +----------> [5] Risk fusion + calibration + rate limiting ---> second-monitor UI
                                 |
                                 v
                       [6] Event timeline (JSON)
                                 |
                                 v
                       [7] Local LLM record writer  --->  [8] Code-based verifier  --->  record
```

All components run on one workstation with a single consumer GPU, with the network disabled.

## 3. Components

| # | Component | Input | Output | Model / method | Licence |
|---|---|---|---|---|---|
| 1 | Frame grabber | video file or HDMI capture | frames at 512x324 | OpenCV | BSD |
| 2 | Segmentation | frame | pupil mask, instrument masks, normalised pupil area | PIDNet (or YOLOX-seg), fine-tuned on CaDIS + Cataract-1K | MIT / Apache-2.0 |
| 3 | Phase recognition | ResNet-50 features over 20 s window | phase label + confidence per second | ResNet-50 (torchvision, BSD) + GRU; median-filtered | BSD |
| 4a | Pupil-constriction detector | normalised pupil-area time series | flag + confidence | gradient-boosted classifier on windowed features, threshold learned on Cataract-1K pupil-reaction subset | Apache-2.0 |
| 4b | Radial-fold classifier | 2-4 s clips during capsulorhexis | flag + confidence + one-line explanation | 3D-CNN baseline; optional Qwen3-VL LoRA for explanations | Apache-2.0 |
| 4c | Phaco-duration rule | running phaco-phase duration | flag at 85th / 95th percentile | empirical distribution from public videos; MNGHA prior if tabular data released | n/a |
| 5 | Risk fusion | detector outputs (+ optional pre-op prior) | routine / rising / high, calibrated probability | logistic fusion with isotonic calibration; 30 s per-detector rate limit | n/a |
| 6 | Timeline | events + phases with timestamps | JSON record | schema in `configs/timeline_schema.json` | n/a |
| 7 | Record writer | timeline JSON only (never video) | structured complication-risk record | Qwen3 (4B or 8B, Apache-2.0), LoRA on synthetic JSON-to-record pairs, fixed template | Apache-2.0 |
| 8 | Verifier | record + timeline | pass/fail per statement; factual-accuracy % | regex/field matching | n/a |

## 4. Data flow and privacy boundaries

* **Environment A (team hardware / rented GPU):** public datasets only (Cataract-1K, CaDIS, Cataract-101, Cataract-LMM, Tongren-Zonular-Video). Training and quantitative validation happen here.
* **Environment B (MNGHA sanctioned environment):** institutional data only. Same code, `--offline`. Used for adaptation (fine-tuning last layers on 20-50 local videos if released), fusion calibration, qualitative checks, and the live demo. Only metrics and plots leave; adapted weights are left inside unless the organisers permit export. Temporary files purged at the end (rule 4.3).
* No component makes a network call. Rule 5.2 is satisfied structurally, not by policy.

## 5. Training and evaluation protocol

* Video-level splits: train 70% / validation 15% / test 15% within Cataract-1K, CaDIS, Cataract-LMM; **Cataract-101 is fully held out as the external test set.**
* Thresholds and calibration are fitted on validation, reported on test and external.
* Metrics: pupil Dice/IoU; phase accuracy, per-phase F1, confusion matrix, sequence-order check; per-detector precision, recall, F1, lead time (s), false alerts per case; phaco-duration error (s); record factual accuracy (% verified statements); latency per frame (ms) and end-to-end fps.
* Targets set against published benchmarks: phase accuracy ~90%; pupil Dice 88-90%; radial-fold AUROC ~0.9; >=15 fps on a laptop GPU.

## 6. Interface

Second-monitor web dashboard (FastAPI + static front end, served on localhost): current phase, pupil-area trace, three marker indicators with time since last flag, risk band, event log, end-of-case record. Nothing is drawn inside the surgeon's oculars. All UI text is observational.

## 7. Hardware and runtime

* Development: AMD Radeon RX 7800 XT (16 GB, ROCm) and free-tier cloud GPUs; NVIDIA for optional 7B fine-tunes.
* Inference: any 8-16 GB GPU; models exported to ONNX; device-agnostic PyTorch fallback.
* Language model served locally via Ollama or vLLM, bound to localhost.

## 8. Known limitations

* Radial-fold detector is trained on Chinese and Singaporean video; transfer to Gulf eyes is a stated validation question, not a claim.
* Rare complications (PCR itself) are not detected; the system reports risk awareness, not diagnosis (published PCR detection F1 ~0.61).
* Pupil-area normalisation depends on limbus visibility; heavy glare degrades segmentation.
* Saudi case-mix tuning is a design claim until validated on MNGHA video.

## 9. Provenance

Pre-built on public data before the event and declared at registration (rule 3.2). Tag `pre-event-v1` separates pre-event and in-event work. Generative AI use is disclosed in `docs/AI_DISCLOSURE.md`.
