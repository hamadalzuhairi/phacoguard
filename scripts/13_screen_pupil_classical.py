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
from phacoguard.detectors.pupil_measured import (  # noqa: E402
    compare_raw_and_filtered, deepest_drop, read_area_csv,
)

FIELDS = [
    "frame_index", "t_s", "pupil_px", "limbus_px", "normalised_pupil_area",
    "circularity", "fill_ratio", "specular_fraction", "instrument_overlap", "clean", "reject_reason",
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
    results = []
    for v in videos:
        t0 = time.time()
        csv_path = out_dir / f"{v.stem}.csv"
        n, clean = screen(v, csv_path, args.fps)
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


def screen(video: Path, out: Path, fps: float) -> tuple[int, int]:
    """Measure one video and write the CSV. Returns (frames, clean_frames)."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(src_fps / fps)))

    ok, first = cap.read()
    if not ok:
        raise SystemExit(f"cannot read first frame of {video}")
    limbus = pc.find_limbus(first)

    rows, idx, n = [], 0, 0
    while idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        m = pc.measure_frame(frame, limbus)
        reason = pc.judge(m, limbus.area_px)
        rows.append({
            "frame_index": n, "t_s": round(idx / src_fps, 3),
            "pupil_px": m["pupil_px"], "limbus_px": limbus.area_px,
            "normalised_pupil_area": round(m["pupil_px"] / limbus.area_px, 6) if limbus.area_px else 0.0,
            "circularity": m["circularity"], "fill_ratio": m.get("fill_ratio", 0.0),
            "specular_fraction": m["specular_fraction"],
            "instrument_overlap": 0.0, "clean": int(not reason), "reject_reason": reason,
        })
        n += 1
        idx += step
    cap.release()

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(rows), sum(r["clean"] for r in rows)


if __name__ == "__main__":
    main()
