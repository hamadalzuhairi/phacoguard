"""Measure pupil area over a cataract video with SAM 2 zero-shot tracking.

This produces a *measurement*, not an expert annotation. Nothing here reads a
clinician label, and the indicator it feeds is labelled in the UI as
"measured pupil-area decrease, ungated (no phase labels for this case)".

Method
------
1. Sample frames at `--fps`. The detector works on a 30 s window with a 10 s
   persistence rule, so a couple of frames per second is ample.
2. On the first sampled frame, place one point prompt inside the pupil, located
   by a Hough-circle prior and overridable with `--pupil-xy`. The prompt is a
   geometric starting point, not a clinical judgement.
3. Propagate that mask with the SAM 2 video predictor.
4. Per frame, judge whether the pupil reading is *clean* or *occluded*:
   - **circularity** `4*pi*A/P^2` — an instrument crossing the pupil makes the
     mask non-round, so a sharp fall flags occlusion;
   - **specular fraction** — the share of pupil-mask pixels that are metallic
     bright, a cheap proxy for an instrument lying over the pupil;
   - **instrument overlap** — only with `--track-instrument`, which adds a second
     SAM 2 object prompted on the brightest metallic blob in frame 0.
5. Write raw and occlusion-filtered traces side by side so the difference is
   visible rather than assumed.

Why the instrument is not tracked by default: a second SAM 2 object roughly
triples cost (measured 3.27 -> 10.5 s/frame on this CPU), and it can only be
prompted on an instrument already visible in frame 0, so it misses instruments
that enter later. Circularity plus the specular fraction catch those and cost
nothing. `--track-instrument` remains available for the workstation.

Network access (checkpoint download) lives here in `scripts/`, never in
`src/phacoguard` (CLAUDE.md rule 2). SAM 2 is Apache-2.0 (rule 3).

    python scripts/12_track_pupil_sam2.py --video data/cataract1k/pupil_reaction/case_709.mp4
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phacoguard.runlock import RunLock, RunLockBusy  # noqa: E402

CHECKPOINTS = {
    "tiny": ("sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml",
             "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"),
    "small": ("sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml",
              "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"),
}
PUPIL_ID, INSTRUMENT_ID = 1, 2

MIN_CIRCULARITY = 0.80      # a clean pupil mask is close to round
MAX_SPECULAR_FRACTION = 0.12
MAX_INSTRUMENT_OVERLAP = 0.05
SPECULAR_LEVEL = 230

FIELDS = [
    "frame_index", "t_s", "pupil_px", "limbus_px", "normalised_pupil_area",
    "circularity", "specular_fraction", "instrument_overlap", "clean", "reject_reason",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=float, default=2.0, help="sampling rate")
    ap.add_argument("--width", type=int, default=512, help="downscale longer edge to at most this")
    ap.add_argument("--model", choices=list(CHECKPOINTS), default="tiny")
    ap.add_argument("--weights-dir", default="weights")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--pupil-xy", type=int, nargs=2, default=None)
    ap.add_argument("--track-instrument", action="store_true",
                    help="add a second SAM 2 object for the instrument (about 3x slower)")
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    video = Path(args.video)
    out = Path(args.out) if args.out else Path("outputs/pupil") / f"{video.stem}.csv"
    work = Path(args.work_dir) if args.work_dir else Path("outputs/pupil/_frames") / video.stem

    work.mkdir(parents=True, exist_ok=True)
    try:
        lock = RunLock(work).acquire()
    except RunLockBusy as busy:
        raise SystemExit(str(busy))

    try:
        _track_one(video, out, work, args, lock)
    finally:
        lock.release()


def _track_one(video: Path, out: Path, work: Path, args, lock) -> None:
    ckpt = _ensure_checkpoint(args.model, Path(args.weights_dir), args.offline)
    n, size = _sample_frames(video, work, args.fps, args.max_frames, args.width)
    print(f"{video.name}: {n} sampled frames at {args.fps} fps, {size[0]}x{size[1]}", flush=True)

    first = cv2.imread(str(work / "00000.jpg"))
    auto_xy, limbus_px = locate_pupil(first)
    pupil_xy = tuple(args.pupil_xy) if args.pupil_xy else auto_xy
    instrument_xy = locate_instrument(first) if args.track_instrument else None
    print(f"  pupil prompt {pupil_xy}; limbus reference {limbus_px} px"
          + (f"; instrument prompt {instrument_xy}" if instrument_xy else ""), flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    rows, seconds = track(work, ckpt, CHECKPOINTS[args.model][1], pupil_xy, limbus_px,
                          instrument_xy, out, lock)
    clean = sum(1 for r in rows if r["clean"])
    print(f"  {clean}/{len(rows)} frames clean ({100*clean/max(len(rows),1):.0f}%)")
    print(f"  wrote {out}")
    print(f"TIMING {video.stem} frames={len(rows)} seconds={seconds:.0f} "
          f"per_frame={seconds/max(len(rows),1):.2f}", flush=True)


def locate_pupil(frame: np.ndarray) -> tuple[tuple[int, int], int]:
    """Pupil click point and the limbus disc area, from a Hough-circle prior."""
    g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (9, 9), 2)
    h, w = g.shape
    circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(h, w) // 4,
                               param1=100, param2=30,
                               minRadius=int(min(h, w) * 0.08), maxRadius=int(min(h, w) * 0.45))
    if circles is not None:
        c = sorted(np.round(circles[0]).astype(int),
                   key=lambda c: abs(c[0] - w // 2) + abs(c[1] - h // 2))
        cx, cy, r = c[0]
        return (int(cx), int(cy)), int(np.pi * r * r)
    return (w // 2, h // 2), int(np.pi * (min(h, w) * 0.35) ** 2)


def locate_instrument(frame: np.ndarray) -> tuple[int, int] | None:
    """Centroid of the largest metallic-bright blob away from the frame centre."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(g, SPECULAR_LEVEL, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    best, best_area = None, 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > best_area:
            best_area, best = stats[i, cv2.CC_STAT_AREA], (int(cents[i][0]), int(cents[i][1]))
    return best if best_area > 30 else None


def mask_metrics(mask: np.ndarray, frame: np.ndarray) -> tuple[float, float]:
    """Circularity of the mask, and the share of its pixels that are metallic bright."""
    m = mask.astype(np.uint8)
    area = int(m.sum())
    if area == 0:
        return 0.0, 0.0
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circ = 0.0
    if contours:
        c = max(contours, key=cv2.contourArea)
        per = cv2.arcLength(c, True)
        if per > 0:
            circ = float(4 * np.pi * cv2.contourArea(c) / (per * per))
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    specular = float((grey[m.astype(bool)] > SPECULAR_LEVEL).sum()) / area
    return min(circ, 1.0), specular


def track(frames_dir: Path, ckpt: Path, cfg: str, pupil_xy, limbus_px: int,
          instrument_xy, out: Path, lock=None) -> tuple[list[dict], float]:
    """Rows are flushed to `out` as they are produced, so a killed run keeps its work."""
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_sam2_video_predictor(cfg, str(ckpt), device=device)
    times = _read_times(frames_dir)
    frames = sorted(frames_dir.glob("*.jpg"))

    t0 = time.time()
    fh = open(out, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    writer.writeheader()
    with torch.inference_mode():
        state = predictor.init_state(video_path=str(frames_dir))
        predictor.add_new_points_or_box(
            inference_state=state, frame_idx=0, obj_id=PUPIL_ID,
            points=np.array([pupil_xy], dtype=np.float32), labels=np.array([1], dtype=np.int32))
        if instrument_xy is not None:
            predictor.add_new_points_or_box(
                inference_state=state, frame_idx=0, obj_id=INSTRUMENT_ID,
                points=np.array([instrument_xy], dtype=np.float32),
                labels=np.array([1], dtype=np.int32))

        rows, done = [], 0
        for idx, obj_ids, logits in predictor.propagate_in_video(state):
            masks = {oid: (logits[i][0] > 0.0).cpu().numpy() for i, oid in enumerate(obj_ids)}
            pupil_mask = masks.get(PUPIL_ID)
            if pupil_mask is None:
                continue
            frame = cv2.imread(str(frames[idx]))
            area = int(pupil_mask.sum())
            circ, spec = mask_metrics(pupil_mask, frame)

            overlap = 0.0
            inst = masks.get(INSTRUMENT_ID)
            if inst is not None and area:
                overlap = float(np.logical_and(inst, pupil_mask).sum()) / area

            reason = ""
            if area == 0:
                reason = "no pupil mask"
            elif circ < MIN_CIRCULARITY:
                reason = f"circularity {circ:.2f}"
            elif spec > MAX_SPECULAR_FRACTION:
                reason = f"specular {spec:.2f}"
            elif overlap > MAX_INSTRUMENT_OVERLAP:
                reason = f"instrument overlap {overlap:.2f}"

            row = {
                "frame_index": idx, "t_s": round(times[idx], 3),
                "pupil_px": area, "limbus_px": limbus_px,
                "normalised_pupil_area": round(area / limbus_px, 6) if limbus_px else 0.0,
                "circularity": round(circ, 4), "specular_fraction": round(spec, 4),
                "instrument_overlap": round(overlap, 4),
                "clean": int(not reason), "reject_reason": reason,
            }
            rows.append(row)
            writer.writerow(row)
            fh.flush()
            done += 1
            if done % 50 == 0:
                el = time.time() - t0
                print(f"    {done}/{len(frames)} frames  {el:.0f}s  {el/done:.2f} s/frame", flush=True)
                if lock is not None:
                    lock.heartbeat()
    fh.close()
    return rows, time.time() - t0


def _sample_frames(video: Path, work: Path, fps: float, cap_n: int | None,
                   max_width: int) -> tuple[int, tuple[int, int]]:
    work.mkdir(parents=True, exist_ok=True)
    for old in work.glob("*.jpg"):
        old.unlink()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(src_fps / fps)))
    idx = n = 0
    size = (0, 0)
    times = []
    while idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] > max_width:
            s = max_width / frame.shape[1]
            frame = cv2.resize(frame, (max_width, int(frame.shape[0] * s)))
        cv2.imwrite(str(work / f"{n:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        times.append(idx / src_fps)
        size = (frame.shape[1], frame.shape[0])
        n += 1
        if cap_n and n >= cap_n:
            break
        idx += step
    cap.release()
    (work / "times.csv").write_text("\n".join(f"{i},{t}" for i, t in enumerate(times)), encoding="utf-8")
    return n, size


def _read_times(frames_dir: Path) -> dict[int, float]:
    out = {}
    for line in (frames_dir / "times.csv").read_text(encoding="utf-8").splitlines():
        i, t = line.split(",")
        out[int(i)] = float(t)
    return out


def _ensure_checkpoint(model: str, weights_dir: Path, offline: bool) -> Path:
    name, _, url = CHECKPOINTS[model]
    weights_dir.mkdir(parents=True, exist_ok=True)
    dest = weights_dir / name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    if offline:
        raise SystemExit(f"--offline and {dest} is missing")
    print(f"fetching {name} ...")
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    tmp.replace(dest)
    return dest


if __name__ == "__main__":
    main()
