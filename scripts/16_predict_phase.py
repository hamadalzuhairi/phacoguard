"""Predict surgical steps for a video with the quick phase model.

Output is explicitly provisional: every consumer labels it "phases predicted by
a quick model (n=N training cases, test accuracy X%); full model pending".

Steps whose test F1 falls below `--merge-below` are merged into their following
neighbour **for the phase bar only**. All ten steps keep their own row in the
step list; the caption names the steps the model cannot yet distinguish.

    python scripts/16_predict_phase.py --videos case_742 case_800
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from phacoguard.data.phase_map import PHASES, SURGICAL_STEPS, TRANSITION  # noqa: E402
from phacoguard.runlock import RunLock, RunLockBusy  # noqa: E402

MERGE_TARGET = {"ovd_injection": "capsulorhexis", "capsule_polishing": "cortex_removal",
                "wound_closure": "ovd_removal"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--videos-dir", default="data/cataract1k/pupil_reaction")
    ap.add_argument("--model-dir", default="outputs/phase")
    ap.add_argument("--out-dir", default="outputs/phase/predictions")
    ap.add_argument("--fps", type=float, default=0.5)
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--smooth", type=int, default=3, help="median filter over N samples")
    ap.add_argument("--merge-below", type=float, default=0.60,
                    help="merge a step into its neighbour for the bar when test F1 is below this")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        lock = RunLock(out).acquire()
    except RunLockBusy as busy:
        raise SystemExit(str(busy))
    try:
        _run(args, out, lock)
    finally:
        lock.release()


def _run(args, out: Path, lock) -> None:
    import torch
    from torch import nn

    report = json.loads((Path(args.model_dir) / "phase_report.json").read_text(encoding="utf-8"))
    f1 = report["per_class_f1"]
    weak = [s for s in SURGICAL_STEPS if f1.get(s, 1.0) < args.merge_below]
    merges = {s: MERGE_TARGET.get(s, TRANSITION) for s in weak}
    print(f"quick model: n={report['n_train_cases']} training cases, "
          f"test accuracy {report['test_frame_accuracy']*100:.1f}%")
    print(f"steps below F1 {args.merge_below} (merged for the bar only): "
          f"{ {s: f'{f1[s]:.2f} -> {t}' for s, t in merges.items()} or 'none'}\n")

    ckpt = torch.load(Path(args.model_dir) / "quick_phase_gru.pt", weights_only=False)
    gru = nn.GRU(512, 128, num_layers=1, batch_first=True, bidirectional=True)
    head = nn.Linear(256, len(PHASES))
    gru.load_state_dict(ckpt["gru"]); head.load_state_dict(ckpt["head"])
    gru.eval(); head.eval()
    torch.set_grad_enabled(False)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    trainer = __import__("importlib").import_module("importlib").import_module  # noqa
    from runpy import run_path
    mod = run_path(str(Path(__file__).with_name("11_train_phase.py")), run_name="_phase_lib")
    encoder = mod["_encoder"]()
    sample = mod["_sample"]

    for case in args.videos:
        video = Path(args.videos_dir) / f"{case}.mp4"
        if not video.exists():
            raise SystemExit(f"missing {video}")
        frames, times = sample(video, args.fps, args.size)
        if not frames:
            raise SystemExit(f"no frames sampled from {video}")
        x = _encode(frames, encoder)
        logits = head(gru(torch.from_numpy(x).unsqueeze(0))[0].squeeze(0))
        pred = logits.argmax(1).numpy()
        pred = _median_filter(pred, args.smooth)
        intervals = _intervals(pred, times, args.fps)
        payload = {
            "case_id": case,
            "source": "quick phase model",
            "n_train_cases": report["n_train_cases"],
            "test_frame_accuracy": report["test_frame_accuracy"],
            "merged_for_bar": merges,
            "fps": args.fps,
            "intervals": intervals,
        }
        path = out / f"{case}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        lock.heartbeat()
        steps = [i["phase"] for i in intervals if i["phase"] != TRANSITION]
        print(f"{case}: {len(times)} samples, {len(intervals)} intervals, "
              f"{len(set(steps))} distinct steps -> {path}")
        for i in intervals:
            if i["phase"] != TRANSITION:
                print(f"    {i['start_s']:7.1f}-{i['end_s']:7.1f}s  {i['phase']}")


def _encode(frames, encoder):
    import torch
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    b = (np.stack(frames).astype(np.float32) / 255.0 - mean) / std
    t = torch.from_numpy(b.transpose(0, 3, 1, 2))
    return np.concatenate([encoder(t[i:i + 32]).numpy() for i in range(0, len(t), 32)]).astype(np.float32)


def _median_filter(pred: np.ndarray, n: int) -> np.ndarray:
    if n < 2:
        return pred
    half = n // 2
    out = pred.copy()
    for i in range(len(pred)):
        window = pred[max(0, i - half): i + half + 1]
        vals, counts = np.unique(window, return_counts=True)
        out[i] = vals[counts.argmax()]
    return out


def _intervals(pred: np.ndarray, times: list[float], fps: float) -> list[dict]:
    step_s = 1.0 / fps
    out: list[dict] = []
    for i, p in enumerate(pred):
        phase = PHASES[int(p)]
        start, end = times[i], times[i] + step_s
        if out and out[-1]["phase"] == phase:
            out[-1]["end_s"] = round(end, 2)
        else:
            out.append({"phase": phase, "start_s": round(start, 2), "end_s": round(end, 2)})
    return out


if __name__ == "__main__":
    main()
