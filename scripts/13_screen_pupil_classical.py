"""Screen many videos with the classical pupil tracker and rank them by drop depth.

Fast triage: seconds to minutes per video, so every candidate can be measured and
ranked. SAM 2 then confirms only the shortlist.

    python scripts/13_screen_pupil_classical.py --dir data/cataract1k/pupil_reaction
    python scripts/13_screen_pupil_classical.py --dir ... --rank --top 5

Output per video matches the SAM 2 tracker's CSV schema exactly, so the same
detector, occlusion filter and persistence rule read either one.

No network access.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402

from phacoguard.detectors import pupil_classical as pc  # noqa: E402
from phacoguard.runlock import RunLock, RunLockBusy  # noqa: E402
from phacoguard.detectors.pupil_measured import (  # noqa: E402
    compare_raw_and_filtered, deepest_drop, read_area_csv,
)

FIELDS = [
    "frame_index", "t_s", "pupil_px", "limbus_px", "normalised_pupil_area",
    "circularity", "fill_ratio", "specular_fraction", "instrument_overlap",
    "mode", "limbus_r", "clean", "reject_reason",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="data/cataract1k/pupil_reaction")
    ap.add_argument("--out-dir", default="outputs/pupil/classical")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--rank", action="store_true", help="rank videos and write a shortlist")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--only", nargs="*", default=None, help="limit to these stems")
    args = ap.parse_args()

    videos = sorted(Path(args.dir).glob("*.mp4"))
    if args.only:
        videos = [v for v in videos if v.stem in set(args.only)]
    if not videos:
        raise SystemExit(f"no videos under {args.dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        lock = RunLock(out_dir).acquire()
    except RunLockBusy as busy:
        raise SystemExit(str(busy))
    if lock.took_over_stale:
        print(f"note: took over an abandoned lock from pid "
              f"{lock.took_over_stale.get('pid')}", flush=True)
    try:
        _run(videos, out_dir, args, lock)
    finally:
        lock.release()


def _run(videos, out_dir: Path, args, lock) -> None:
    results = []
    for v in videos:
        t0 = time.time()
        csv_path = out_dir / f"{v.stem}.csv"
        n, clean, switches, lstats = screen(v, csv_path, args.fps)
        el = time.time() - t0
        samples = read_area_csv(csv_path)
        cmp_ = compare_raw_and_filtered(samples) if samples else {"filtered_events": [], "raw_events": []}
        events = cmp_["filtered_events"]
        # Rank by the deepest drop that actually passed every rule, not by the raw
        # series minimum: an unvetted minimum is what surfaced the start-of-video
        # artefacts in the first pass.
        depth = max((e.max_drop_fraction for e in events), default=0.0)
        results.append({
            "video": v.stem, "frames": n, "clean": clean,
            "clean_fraction": round(clean / n, 3) if n else 0.0,
            "deepest_filtered_drop": round(depth, 4),
            "unvetted_deepest_drop": round(deepest_drop(samples), 4) if samples else 0.0,
            "n_filtered_events": len(events),
            "n_raw_events": len(cmp_["raw_events"]),
            "first_event_t_s": round(events[0].t_s, 1) if events else None,
            "mode_switches": [sw.as_dict() for sw in switches],
            "limbus_accepted": lstats["limbus_accepted"],
            "limbus_rejected": lstats["limbus_rejected"],
            "seconds": round(el, 1),
        })
        print(f"{v.stem:14s} {n:5d} frames  {clean:5d} clean  "
              f"deepest drop {depth*100:5.1f}%  events {len(cmp_['filtered_events'])}  "
              f"{el:6.1f}s", flush=True)

    summary = out_dir / "screen_summary.json"
    summary.write_text(json.dumps(results, indent=2), encoding="utf-8")
    total = sum(r["seconds"] for r in results)
    print(f"\n{len(results)} videos in {total:.0f}s ({total/max(len(results),1):.1f}s per video)")
    print(f"summary -> {summary}")

    if args.rank:
        ranked = sorted(results, key=lambda r: -r["deepest_filtered_drop"])
        shortlist = ranked[: args.top]
        label = (f"ranked by classical screener across {len(results)}; "
                 f"SAM 2-confirmed on 0")
        path = out_dir / "shortlist.json"
        path.write_text(json.dumps({"label": label, "n_screened": len(results),
                                    "shortlist": shortlist}, indent=2), encoding="utf-8")
        print(f"\n{label}")
        for i, r in enumerate(shortlist, 1):
            print(f"  {i}. {r['video']:14s} drop {r['deepest_filtered_drop']*100:.1f}%  "
                  f"clean {r['clean_fraction']*100:.0f}%")
        print(f"shortlist -> {path}")


def screen(video: Path, out: Path, fps: float) -> tuple[int, int, list, dict]:
    """Measure one video. Returns (frames, clean_frames, mode_switches, limbus_stats)."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(src_fps / fps)))

    tracker = pc.LimbusTracker()
    mode, switches, next_decision = None, [], -1.0
    rows, idx, n = [], 0, 0

    while idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        t_s = idx / src_fps
        limbus = tracker.update(frame, t_s)

        if t_s >= next_decision:
            window = _window_frames(cap, idx, total, step, src_fps, pc.MODE_WINDOW_S)
            new_mode, contrast = pc.decide_mode(window or [frame], limbus, previous=mode)
            if mode is not None and new_mode is not mode:
                switches.append(pc.ModeSwitch(t_s, mode, new_mode, contrast))
            mode = new_mode
            next_decision = t_s + pc.MODE_WINDOW_S
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            cap.read()

        m = pc.measure_frame(frame, limbus, mode)
        reason = pc.judge(m, limbus.area_px)
        rows.append({
            "frame_index": n, "t_s": round(t_s, 3),
            "pupil_px": m["pupil_px"], "limbus_px": limbus.area_px,
            "normalised_pupil_area": round(m["pupil_px"] / limbus.area_px, 6) if limbus.area_px else 0.0,
            "circularity": m["circularity"], "fill_ratio": m.get("fill_ratio", 0.0),
            "specular_fraction": m["specular_fraction"], "instrument_overlap": 0.0,
            "mode": m.get("mode", ""), "limbus_r": limbus.r,
            "clean": int(not reason), "reject_reason": reason,
        })
        n += 1
        idx += step
    cap.release()

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    stats = {"limbus_accepted": tracker.n_accepted, "limbus_rejected": tracker.n_rejected}
    return len(rows), sum(r["clean"] for r in rows), switches, stats


def _window_frames(cap, idx: int, total: int, step: int, src_fps: float,
                   window_s: float, k: int = 3) -> list:
    """A few frames spread across the coming window, for the mode decision."""
    span = int(window_s * src_fps)
    out = []
    for j in range(k):
        at = idx + int(span * j / k)
        if at >= total:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, at)
        ok, f = cap.read()
        if ok:
            out.append(f)
    return out


if __name__ == "__main__":
    main()
