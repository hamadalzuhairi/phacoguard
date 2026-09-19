# Working prototype demonstration

*Hackathon deliverable (rule 3.3): live demo or recorded video. Plan: live demo with a recorded fallback.*

## What the judges see (3 minutes)

1. **Setup (0:00-0:20).** Laptop with PhacoGuard running offline; second monitor (or split screen) shows the dashboard. Network visibly disabled (airplane mode on screen).
2. **Routine case (0:20-1:00).** A public phaco video plays through the pipeline. Phase bar advances; pupil trace flat; risk band stays *routine*; no alerts. Point: "quiet on routine cases" (false alerts per case on screen).
3. **Complex case (1:00-2:20).** A public case with pupil constriction / radial folds:
   * During capsulorhexis the radial-fold indicator lights with the model's one-line explanation.
   * During phaco the pupil trace dips; the constriction indicator lights; band moves to *rising*.
   * Phaco timer crosses the 85th percentile; band moves to *high*.
   * Emphasise the rate limiting: indicators, not a siren.
4. **Case end (2:20-2:50).** Timeline JSON appears; the local LLM writes the complication-risk record; the verifier marks every statement as checked.
5. **Numbers (2:50-3:00).** One slide: external-test metrics; if MNGHA video was released, before/after adaptation on Saudi eyes.

## Fallback

`outputs/demo/phacoguard_demo_v1.mp4`: screen recording of steps 2-4 on public video, recorded before the event. Played if the venue machine, display, or environment fails. Never recorded inside the MNGHA environment unless the organisers permit it; never contains institutional data.

## Live demo checklist

- [ ] Weights cached locally; `python scripts/30_run_pipeline.py --video demo/routine.mp4 --offline` runs end to end with Wi-Fi off
- [ ] Second monitor or window layout tested at venue resolution
- [ ] Two public demo videos chosen (routine; complex) and pre-decoded
- [ ] Fallback video on the laptop and on a USB drive
- [ ] Metrics slide exported as PNG in case the deck fails
- [ ] All UI text is observational (no "do X" phrases)

## Recording the fallback video

```bash
python scripts/30_run_pipeline.py --video demo/complex.mp4 --offline --record outputs/demo/phacoguard_demo_v1.mp4
```
