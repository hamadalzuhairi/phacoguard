"""Diagnose why the classical screener rejects most frames.

Produces, per requested case:

* a contact sheet of randomly chosen rejected frames and accepted frames, each
  annotated with its reject reason, so the filter's decisions can be looked at
  rather than inferred;
* the Otsu threshold on the a\\* channel over time, to show whether segmentation
  is stable or jumping between regimes.

Diagnostic only: it changes no thresholds.

    python scripts/15_diagnose_rejections.py --cases case_8441 case_800 case_8167
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from phacoguard.detectors import pupil_classical as pc  # noqa: E402

TILE = (256, 192)
COLS = 5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--videos", default="data/cataract1k/pupil_reaction")
    ap.add_argument("--traces", default="outputs/pupil/classical")
    ap.add_argument("--out-dir", default="outputs/pupil/diagnosis")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    for case in args.cases:
        rows = list(csv.DictReader(open(Path(args.traces) / f"{case}.csv", encoding="utf-8-sig")))
        acc = [r for r in rows if not r["reject_reason"].strip()]
        rej = [r for r in rows if r["reject_reason"].strip()]
        print(f"{case}: {len(rows)} frames, {len(acc)} accepted, {len(rej)} rejected")

        for label, pool in (("rejected", rej), ("accepted", acc)):
            pick = rng.sample(pool, min(args.n, len(pool))) if pool else []
            if not pick:
                print(f"   no {label} frames to show")
                continue
            sheet = _contact_sheet(Path(args.videos) / f"{case}.mp4", pick, args.fps, label)
            p = out / f"{case}_{label}.png"
            cv2.imwrite(str(p), sheet)
            print(f"   {label}: {len(pick)} frames -> {p}")

        _threshold_plot(Path(args.videos) / f"{case}.mp4", args.fps, out / f"{case}_otsu.csv")


def _contact_sheet(video: Path, rows: list[dict], fps: float, label: str) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ok, first = cap.read()
    limbus = pc.find_limbus(first)
    tiles = []
    for r in sorted(rows, key=lambda r: float(r["t_s"])):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(r["t_s"]) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        vis = frame.copy()
        cv2.circle(vis, (limbus.cx, limbus.cy), limbus.r, (255, 200, 0), 1)
        m = pc.measure_frame(frame, limbus)
        mask = _mask_of(frame, limbus)
        if mask is not None:
            cont, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cont, -1, (0, 255, 255), 1)
        vis = cv2.resize(vis, TILE)
        reason = r["reject_reason"].strip() or "accepted"
        cv2.rectangle(vis, (0, TILE[1] - 30), (TILE[0], TILE[1]), (0, 0, 0), -1)
        cv2.putText(vis, f"{float(r['t_s']):.0f}s c={float(r['circularity']):.2f}",
                    (4, TILE[1] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        cv2.putText(vis, reason[:34], (4, TILE[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 200, 255), 1)
        tiles.append(vis)
    cap.release()
    while len(tiles) % COLS:
        tiles.append(np.zeros((TILE[1], TILE[0], 3), np.uint8))
    grid = np.vstack([np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)])
    banner = np.zeros((34, grid.shape[1], 3), np.uint8)
    cv2.putText(banner, f"{video.stem} - {label} frames (yellow = pupil mask, blue = Hough limbus)",
                (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return np.vstack([banner, grid])


def _mask_of(frame: np.ndarray, limbus: pc.Limbus) -> np.ndarray | None:
    inside = limbus.mask(frame.shape).astype(bool)
    a = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 1]
    vals = a[inside]
    if vals.size == 0 or vals.max() == vals.min():
        return None
    cut, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    red = ((a >= cut) & inside).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(red, 8)
    if n <= 1:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == best).astype(np.uint8)


def _threshold_plot(video: Path, fps: float, out_csv: Path) -> None:
    """Otsu cut on a* per sampled frame, to show regime stability."""
    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(src_fps / fps)))
    ok, first = cap.read()
    limbus = pc.find_limbus(first)
    inside = None
    rows, idx = [], 0
    while idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        if inside is None:
            inside = limbus.mask(frame.shape).astype(bool)
        a = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 1]
        vals = a[inside]
        cut = 0.0
        if vals.size and vals.max() != vals.min():
            cut, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        rows.append({"t_s": round(idx / src_fps, 2), "otsu_a": round(float(cut), 2),
                     "a_median": float(np.median(vals)), "a_p95": float(np.percentile(vals, 95))})
        idx += step
    cap.release()
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["t_s", "otsu_a", "a_median", "a_p95"])
        w.writeheader()
        w.writerows(rows)
    print(f"   otsu trace -> {out_csv}")


if __name__ == "__main__":
    main()
