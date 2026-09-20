"""Run the pipeline on a video offline; write timeline JSON, record, and demo video.

`--mock` implements HANDOFF.md section 2a: the dashboard runs over real public
footage while the indicators fire at moments taken from expert dataset labels.
No model runs in this mode and no threshold is chosen by the team.

    python scripts/30_run_pipeline.py --mock --data-root data/cataract1k

The model-driven mode (no --mock) is not implemented yet; it replaces the event
sources and reuses the same timeline, dashboard, and recorder.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from phacoguard.mock import cataract1k, pupil_segment, sources, timeline as tl  # noqa: E402
from phacoguard.render.recorder import DemoWriter  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--phase-map", default="configs/phase_map.yaml")
    ap.add_argument("--mock", action="store_true", help="section 2a label-driven demo")
    ap.add_argument("--data-root", default="data/cataract1k")
    ap.add_argument("--out", default="outputs/demo")
    ap.add_argument("--video", default=None, help="model-driven mode only (not implemented)")
    ap.add_argument("--record", default="phacoguard_mock_v1.mp4")
    ap.add_argument("--speed", type=float, default=8.0, help="playback speed of each case")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="cap on case time rendered, in case seconds")
    ap.add_argument("--cases", nargs="*", default=None,
                    help="explicit case ids; default is median and longest phaco case")
    ap.add_argument("--demo-cases", default="outputs/pupil/classical/demo_cases.json",
                    help="pupil-reaction segments to append (empty string to omit)")
    ap.add_argument("--pupil-videos", default="data/cataract1k/pupil_reaction")
    ap.add_argument("--pupil-traces", default="outputs/pupil/classical")
    ap.add_argument("--offline", action="store_true",
                    help="assert no network access (always true in Environment B)")
    args = ap.parse_args()

    if not args.mock:
        raise SystemExit(
            "Only --mock is implemented. The model-driven path arrives after the detectors "
            "are trained (HANDOFF.md section 9)."
        )

    root = Path(args.data_root)
    ann_root, vid_root = root / "annotations", root / "videos"
    if not ann_root.exists():
        raise SystemExit(
            f"no annotations under {ann_root}.\n"
            "Fetch them first:\n"
            "  python scripts/00_download_public_data.py --subset phase --accept-licence"
        )

    pmap = yaml.safe_load(open(args.phase_map, encoding="utf-8"))
    print(f"Reading expert labels under {ann_root} ...")
    dist = sources.phaco_duration_distribution(ann_root, pmap)
    print(f"  {dist['n_cases']} labelled cases   "
          f"p50={dist['p50']:.0f}s  p85={dist['p85']:.0f}s  p95={dist['p95']:.0f}s")

    case_ids = args.cases or [sources.median_case(dist), sources.highest_case(dist)]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    timelines = []
    for case_id in case_ids:
        case_dir = ann_root / case_id
        video = vid_root / f"{case_id}.mp4"
        if not case_dir.exists():
            raise SystemExit(f"no annotations for {case_id} under {ann_root}")
        if not video.exists():
            raise SystemExit(
                f"no video for {case_id}. Fetch it:\n"
                f"  python scripts/00_download_public_data.py --subset phase "
                f"--accept-licence --videos {case_id}"
            )
        markers = {
            "pupil_constriction": sources.pupil_constriction(case_id),
            "radial_folds": sources.radial_folds(case_id),
            "prolonged_phaco": sources.prolonged_phaco(
                cataract1k.load_case(case_dir, pmap)[0], case_id, dist
            ),
        }
        t = tl.build(case_dir, video, pmap, dist, markers)
        timelines.append(t)

        path = out_dir / f"timeline_{case_id}.json"
        path.write_text(json.dumps(t.as_dict(), indent=2), encoding="utf-8")
        print(f"\n{case_id}: phaco {t.phaco_duration_s:.0f}s "
              f"(percentile {t.phaco_percentile:.0f}) -> {t.final_state}")
        for name, r in markers.items():
            state = r.status.value if r.status is sources.MarkerStatus.NO_LABELLED_SOURCE \
                else f"{len(r.events)} event(s)"
            print(f"    {name:20s} {state}")
        print(f"    timeline -> {path}")

    # Segment order: a routine phase-labelled case, the prolonged-phaco case,
    # then the pupil-reaction cases. Each states in its own panels why the
    # indicators it cannot drive have no data source.
    if args.demo_cases:
        spec_path = Path(args.demo_cases)
        if not spec_path.exists():
            raise SystemExit(f"no demo case list at {spec_path}; run scripts/13 --rank first")
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        print(f"\n{spec['label']}")
        for entry in spec["cases"]:
            case_id = entry["video"]
            video = Path(args.pupil_videos) / f"{case_id}.mp4"
            trace = Path(args.pupil_traces) / f"{case_id}.csv"
            for needed in (video, trace):
                if not needed.exists():
                    raise SystemExit(f"{case_id}: missing {needed}")
            seg = pupil_segment.build(case_id, video, trace)
            timelines.append(seg)
            path = out_dir / f"timeline_{case_id}.json"
            path.write_text(json.dumps(seg.as_dict(), indent=2), encoding="utf-8")
            merged = sum(e.n_events for e in seg.episodes)
            print(f"  {case_id}: {len(seg.episodes)} episode(s) from {merged} crossing(s) "
                  f"-> {seg.final_state}")
            for ep in seg.episodes:
                print(f"     {ep.start_s:7.1f}-{ep.end_s:7.1f}s  {ep.duration_s:5.0f}s  "
                      f"max drop {ep.max_drop_fraction*100:.0f}%")
            print(f"     timeline -> {path}")

    record = out_dir / args.record
    print(f"\nRendering {record} ...")
    with DemoWriter(record) as writer:
        for t in timelines:
            band = "routine" if t.final_state == "routine" else t.final_state
            if t.episodes:
                detail = (f"{len(t.episodes)} measured constriction episode(s); "
                          f"deepest {max(e.max_drop_fraction for e in t.episodes)*100:.0f}%")
            else:
                detail = (f"labelled phaco time {t.phaco_duration_s:.0f}s "
                          f"(percentile {t.phaco_percentile:.0f})")
            writer.add_card([
                f"{t.dataset}  {t.case_id}",
                f"{t.licence} - {cataract1k.DATASET_CITATION}",
                tl.MOCK_BANNER,
                f"{detail} - expected state: {band}",
            ])
            n = writer.add_case(t, speed=args.speed, max_seconds=args.max_seconds)
            print(f"  {t.case_id}: {n} frames")
    print(f"Wrote {record} ({record.stat().st_size / 1e6:.1f} MB, "
          f"{writer.frames_written} frames)")


if __name__ == "__main__":
    main()
