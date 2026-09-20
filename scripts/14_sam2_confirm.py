"""QUEUED JOB FOR THE WORKSTATION — SAM 2 confirmation of the shortlisted cases.

Not for the ARM64 laptop. SAM 2 measured 3.3 s/frame there for a short clip and
worse for a long one (per-frame cost grows with sequence length), so a single
8-minute video ran past 105 CPU-minutes without finishing. On the Ryzen 7 7700 /
RX 7800 XT machine with ROCm this is minutes per video.

What it does, per shortlisted case:

1. runs `scripts/12_track_pupil_sam2.py` to produce a SAM 2 area trace;
2. compares it with the classical screener trace — Pearson r over time-paired
   clean samples, plus event matching within a tolerance;
3. writes a confirmation report.

Until this has run, every ranking from the screener is labelled
"ranked by classical screener across N; SAM 2 confirmation pending" and every
event it found is provisional.

    python scripts/14_sam2_confirm.py --shortlist outputs/pupil/classical/shortlist.json

Add `--track-instrument` on the workstation: the second SAM 2 object gives a
direct instrument-overlap measurement rather than the circularity proxy, and the
GPU makes the 3x cost affordable.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phacoguard.detectors.pupil_agreement import compare  # noqa: E402
from phacoguard.detectors.pupil_measured import read_area_csv  # noqa: E402

TRACKER = Path(__file__).with_name("12_track_pupil_sam2.py")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shortlist", default="outputs/pupil/classical/shortlist.json")
    ap.add_argument("--videos", default="data/cataract1k/pupil_reaction")
    ap.add_argument("--classical-dir", default="outputs/pupil/classical")
    ap.add_argument("--out-dir", default="outputs/pupil/sam2")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--model", default="tiny", choices=["tiny", "small"])
    ap.add_argument("--track-instrument", action="store_true")
    ap.add_argument("--skip-tracking", action="store_true",
                    help="only re-compare existing SAM 2 traces")
    args = ap.parse_args()

    spec = json.loads(Path(args.shortlist).read_text(encoding="utf-8"))
    cases = [r["video"] for r in spec["shortlist"]]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{spec['label']}\nconfirming {len(cases)} case(s): {', '.join(cases)}\n")

    report = {"source_label": spec["label"], "n_screened": spec["n_screened"], "cases": []}
    for case in cases:
        sam2_csv = out_dir / f"{case}.csv"
        if not args.skip_tracking:
            cmd = [sys.executable, str(TRACKER), "--video", f"{args.videos}/{case}.mp4",
                   "--out", str(sam2_csv), "--fps", str(args.fps), "--model", args.model]
            if args.track_instrument:
                cmd.append("--track-instrument")
            print(f"-> {case}", flush=True)
            t0 = time.time()
            rc = subprocess.call(cmd)
            print(f"   tracker exit {rc} in {time.time()-t0:.0f}s", flush=True)
            if rc != 0:
                report["cases"].append({"video": case, "error": f"tracker exit {rc}"})
                continue

        classical_csv = Path(args.classical_dir) / f"{case}.csv"
        if not (sam2_csv.exists() and classical_csv.exists()):
            report["cases"].append({"video": case, "error": "missing trace"})
            continue

        agreement = compare(read_area_csv(classical_csv), read_area_csv(sam2_csv))
        entry = {"video": case, **agreement.as_dict()}
        report["cases"].append(entry)
        print(f"   r={entry['pearson_r']}  classical {entry['n_classical_events']} events, "
              f"SAM 2 {entry['n_sam2_events']}  -> {entry['verdict']}", flush=True)

    confirmed = sum(1 for c in report["cases"] if c.get("verdict") == "good agreement")
    report["label"] = (f"ranked by classical screener across {spec['n_screened']}; "
                       f"SAM 2-confirmed on {confirmed}")
    path = out_dir / "confirmation_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n{report['label']}\nreport -> {path}")


if __name__ == "__main__":
    main()
