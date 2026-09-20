"""Quick phase model: ResNet-18 features + GRU over the eleven surgical classes.

A deliberately small model whose only job is to give the pupil-reaction videos a
phase bar. It is not the phase model of record — `docs/ARCHITECTURE.md` §3
specifies ResNet-50 features over a 20 s window — and everything it produces is
labelled "predicted by a quick model ... full model pending".

Decoding is sequential OpenCV `grab()`/`retrieve()`, not ffmpeg: measured at
2.9 min for 7,311 s of video, against 3.4 min for per-frame seeking, and it adds
no dependency. ffmpeg is LGPL, outside the permissive set CLAUDE.md rule 3
allows for code.

Splits are by video (rule 5). Network access (torchvision weights) lives here in
`scripts/`, never in `src/phacoguard` (rule 2).

    python scripts/11_train_phase.py --videos 24
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from phacoguard.data.phase_map import PHASES, TRANSITION  # noqa: E402
from phacoguard.data.splits import video_split  # noqa: E402
from phacoguard.mock import cataract1k  # noqa: E402
from phacoguard.runlock import RunLock, RunLockBusy  # noqa: E402

ETA_AFTER = 5
ETA_BUDGET_S = 45 * 60


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", default="data/cataract1k/annotations")
    ap.add_argument("--videos-dir", default="data/cataract1k/videos")
    ap.add_argument("--out-dir", default="outputs/phase")
    ap.add_argument("--videos", type=int, default=24, help="use the N shortest labelled cases")
    ap.add_argument("--fallback-videos", type=int, default=16)
    ap.add_argument("--fps", type=float, default=0.5)
    ap.add_argument("--size", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=17)
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

    pmap = yaml.safe_load(open("configs/phase_map.yaml", encoding="utf-8"))
    cases = _shortest_cases(Path(args.annotations), Path(args.videos_dir), pmap)
    if len(cases) < args.videos:
        print(f"only {len(cases)} cases have both annotations and video on disk", flush=True)
    wanted = cases[: args.videos]
    print(f"{len(wanted)} cases, {sum(d for _, d in wanted):.0f}s of labelled video\n", flush=True)

    encoder = _encoder()
    feats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    started = time.time()
    for i, (case, dur) in enumerate(wanted, 1):
        cache = out / "features" / f"{case}.npz"
        if cache.exists():
            z = np.load(cache)
            feats[case] = (z["x"], z["y"])
        else:
            x, y = _features_for(case, Path(args.videos_dir), Path(args.annotations),
                                 pmap, encoder, args.fps, args.size)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, x=x, y=y)
            feats[case] = (x, y)
        lock.heartbeat()
        el = time.time() - started
        print(f"  [{i:2d}/{len(wanted)}] {case}  {feats[case][0].shape[0]:4d} samples  "
              f"{el:6.0f}s elapsed", flush=True)
        if i == ETA_AFTER:
            projected = el / i * len(wanted)
            print(f"\n  ETA after {ETA_AFTER} videos: {el/i:.1f}s per video -> "
                  f"{projected/60:.1f} min for {len(wanted)} "
                  f"(budget {ETA_BUDGET_S/60:.0f} min)", flush=True)
            if projected > ETA_BUDGET_S and len(wanted) > args.fallback_videos:
                wanted = wanted[: args.fallback_videos]
                print(f"  over budget -> dropping to {args.fallback_videos} videos\n", flush=True)
            else:
                print("  within budget -> continuing\n", flush=True)

    cases_used = [c for c, _ in wanted]
    split = video_split(cases_used, ratios=(0.75, 0.125, 0.125), seed=args.seed)
    print(f"\nsplit by video: train {len(split['train'])} / val {len(split['val'])} / "
          f"test {len(split['test'])}", flush=True)

    report = _train(feats, split, args, lock)
    report["n_train_cases"] = len(split["train"])
    report["cases"] = split
    report["sampling"] = {"fps": args.fps, "size": args.size, "encoder": "resnet18"}
    (out / "phase_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport -> {out / 'phase_report.json'}", flush=True)


def _shortest_cases(ann_root: Path, vid_root: Path, pmap: dict) -> list[tuple[str, float]]:
    rows = []
    for case_dir in cataract1k.find_cases(ann_root):
        if not (vid_root / f"{case_dir.name}.mp4").exists():
            continue
        intervals, _ = cataract1k.load_case(case_dir, pmap)
        rows.append((case_dir.name, max(i.end_s for i in intervals)))
    rows.sort(key=lambda r: r[1])
    return rows


def _encoder():
    import torch
    import torchvision
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Identity()
    m.eval()
    torch.set_grad_enabled(False)
    return m


def _features_for(case: str, vid_root: Path, ann_root: Path, pmap: dict,
                  encoder, fps: float, size: int) -> tuple[np.ndarray, np.ndarray]:
    import torch

    intervals, _ = cataract1k.load_case(ann_root / case, pmap)
    frames, times = _sample(vid_root / f"{case}.mp4", fps, size)
    if not frames:
        return np.zeros((0, 512), np.float32), np.zeros((0,), np.int64)

    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    batch = np.stack(frames).astype(np.float32) / 255.0
    batch = (batch - mean) / std
    tensor = torch.from_numpy(batch.transpose(0, 3, 1, 2))
    chunks = [encoder(tensor[i:i + 32]).numpy() for i in range(0, len(tensor), 32)]
    x = np.concatenate(chunks).astype(np.float32)

    index = {p: i for i, p in enumerate(PHASES)}
    y = np.full(len(times), index[TRANSITION], np.int64)
    for k, t in enumerate(times):
        for iv in intervals:
            if iv.start_s <= t < iv.end_s:
                y[k] = index[iv.phase]
                break
    return x, y


def _sample(video: Path, fps: float, size: int) -> tuple[list[np.ndarray], list[float]]:
    """Sequential decode, retrieving only the frames we keep."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(src / fps)))
    frames, times, i = [], [], 0
    while True:
        if not cap.grab():
            break
        if i % step == 0:
            ok, f = cap.retrieve()
            if ok:
                frames.append(cv2.cvtColor(cv2.resize(f, (size, size)), cv2.COLOR_BGR2RGB))
                times.append(i / src)
        i += 1
    cap.release()
    return frames, times


def _train(feats: dict, split: dict, args, lock) -> dict:
    import torch
    from torch import nn

    torch.manual_seed(args.seed)
    n_classes = len(PHASES)

    def seqs(names):
        return [(torch.from_numpy(feats[n][0]), torch.from_numpy(feats[n][1]))
                for n in names if n in feats and len(feats[n][0])]

    train, val, test = seqs(split["train"]), seqs(split["val"]), seqs(split["test"])
    model = nn.Sequential()
    gru = nn.GRU(512, 128, num_layers=1, batch_first=True, bidirectional=True)
    head = nn.Linear(256, n_classes)
    params = list(gru.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    def forward(x):
        out, _ = gru(x.unsqueeze(0))
        return head(out.squeeze(0))

    def evaluate(data):
        torch.set_grad_enabled(False)
        gru.eval(); head.eval()
        correct = total = 0
        preds, gold = [], []
        for x, y in data:
            p = forward(x).argmax(1)
            correct += int((p == y).sum()); total += len(y)
            preds.append(p.numpy()); gold.append(y.numpy())
        return (correct / max(total, 1),
                np.concatenate(preds) if preds else np.zeros(0, int),
                np.concatenate(gold) if gold else np.zeros(0, int))

    best, best_state, bad = -1.0, None, 0
    for epoch in range(args.epochs):
        torch.set_grad_enabled(True)
        gru.train(); head.train()
        np.random.shuffle(train)
        for x, y in train:
            opt.zero_grad()
            loss = loss_fn(forward(x), y)
            loss.backward()
            opt.step()
        acc, _, _ = evaluate(val)
        if acc > best + 1e-4:
            best, bad = acc, 0
            best_state = ({k: v.clone() for k, v in gru.state_dict().items()},
                          {k: v.clone() for k, v in head.state_dict().items()})
        else:
            bad += 1
        if epoch % 10 == 0 or bad >= args.patience:
            print(f"  epoch {epoch:3d}  val acc {acc:.3f}  best {best:.3f}", flush=True)
        lock.heartbeat()
        if bad >= args.patience:
            print(f"  early stop at epoch {epoch}", flush=True)
            break
    if best_state:
        gru.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])

    acc, preds, gold = evaluate(test)
    f1, support = _per_class_f1(gold, preds, n_classes)
    cm = _confusion(gold, preds, n_classes)
    print(f"\ntest frame accuracy: {acc*100:.1f}%  ({len(gold)} frames)", flush=True)
    print(f"{'step':<20}{'F1':>7}{'support':>9}")
    for i, p in enumerate(PHASES):
        print(f"{p:<20}{f1[i]:7.3f}{support[i]:9d}", flush=True)

    torch.save({"gru": gru.state_dict(), "head": head.state_dict(), "classes": PHASES},
               Path(args.out_dir) / "quick_phase_gru.pt")
    return {
        "test_frame_accuracy": round(acc, 4),
        "val_accuracy": round(best, 4),
        "per_class_f1": {p: round(float(f1[i]), 4) for i, p in enumerate(PHASES)},
        "support": {p: int(support[i]) for i, p in enumerate(PHASES)},
        "confusion_matrix": cm.tolist(),
        "classes": PHASES,
    }


def _per_class_f1(gold, pred, n):
    f1 = np.zeros(n); support = np.zeros(n, int)
    for c in range(n):
        tp = int(((pred == c) & (gold == c)).sum())
        fp = int(((pred == c) & (gold != c)).sum())
        fn = int(((pred != c) & (gold == c)).sum())
        support[c] = tp + fn
        f1[c] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return f1, support


def _confusion(gold, pred, n):
    cm = np.zeros((n, n), int)
    for g, p in zip(gold, pred):
        cm[g, p] += 1
    return cm


if __name__ == "__main__":
    main()
