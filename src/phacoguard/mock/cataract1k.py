"""Readers for the Cataract-1K phase-recognition annotation files.

File layout, as published on Synapse (syn53395146):

    annotations/case_<id>/case_<id>_annotations_phases.csv
    annotations/case_<id>/case_<id>_video.csv
    videos/case_<id>.mp4

Column positions follow the dataset authors' own reader in
`Dataset_codes/phase recognition dataset codes/action_frame_extractor.py`
(github.com/Negin-Ghamsarian/Cataract-1K): the phases file has a header row and
then `[_, phase_name, start_frame, end_frame]`; the video file has a header row
and a single data row whose fifth column is the frame rate.

Reading is deliberately strict. An unexpected column count, a non-numeric frame
index, or a phase name that is not in `configs/phase_map.yaml` raises, because a
silent fallback here would put an unlabelled moment on screen.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

DATASET = "cataract1k"
DATASET_LICENCE = "CC BY 4.0"
DATASET_CITATION = (
    "Ghamsarian et al., Cataract-1K, Scientific Data 11, 373 (2024); Synapse syn52540135"
)


@dataclass(frozen=True)
class PhaseInterval:
    """One expert-labelled phase interval, in seconds."""

    raw_name: str
    phase: str
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def read_fps(video_csv: Path) -> float:
    """Frame rate from `case_<id>_video.csv` (fifth column of the single data row)."""
    rows = _rows(video_csv)
    if not rows:
        raise ValueError(f"{video_csv}: no data row after the header")
    last = rows[-1]
    if len(last) < 5:
        raise ValueError(f"{video_csv}: expected >=5 columns, found {len(last)}: {last!r}")
    fps = float(last[4])
    if not fps > 0:
        raise ValueError(f"{video_csv}: frame rate must be positive, found {fps}")
    return fps


def read_phase_intervals(phases_csv: Path, fps: float, pmap: dict) -> list[PhaseInterval]:
    """Expert phase intervals, converted from frame indices to seconds.

    `pmap` is the parsed `configs/phase_map.yaml`; every raw phase name must be
    present under its `cataract1k` key or this raises, naming the missing entry.
    """
    from phacoguard.data.phase_map import to_phacoguard

    out: list[PhaseInterval] = []
    for n, row in enumerate(_rows(phases_csv), start=2):
        if len(row) < 4:
            raise ValueError(f"{phases_csv} line {n}: expected >=4 columns, found {row!r}")
        raw = row[1].strip()
        try:
            # Columns 5 and 6 (`sec`, `endSec`) carry the dataset's own seconds; prefer them
            # over dividing frames by fps, and fall back to frames when they are absent.
            if len(row) >= 6 and row[4].strip() and row[5].strip():
                start_s, end_s = float(row[4]), float(row[5])
            else:
                start_s, end_s = float(row[2]) / fps, float(row[3]) / fps
        except ValueError as exc:
            raise ValueError(f"{phases_csv} line {n}: non-numeric time in {row!r}") from exc
        if end_s < start_s:
            raise ValueError(f"{phases_csv} line {n}: end {end_s} precedes start {start_s}")
        out.append(
            PhaseInterval(
                raw_name=raw,
                phase=to_phacoguard(DATASET, raw, pmap),
                start_s=start_s,
                end_s=end_s,
            )
        )
    out.sort(key=lambda i: i.start_s)
    return out


def phaco_duration_s(intervals: list[PhaseInterval]) -> float:
    """Total labelled phaco time for one case (a case may have several phaco intervals)."""
    return sum(i.duration_s for i in intervals if i.phase == "phaco")


def case_dir_id(case_dir: Path) -> str:
    """`.../case_4687` -> `case_4687`."""
    return case_dir.name


def find_cases(annotations_root: Path) -> list[Path]:
    """Every `case_*` directory that holds both required CSVs, sorted by case number."""
    cases = [
        d
        for d in sorted(annotations_root.glob("case_*"))
        if d.is_dir()
        and (d / f"{d.name}_annotations_phases.csv").exists()
        and (d / f"{d.name}_video.csv").exists()
    ]
    return sorted(cases, key=lambda d: _case_number(d.name))


def load_case(case_dir: Path, pmap: dict) -> tuple[list[PhaseInterval], float]:
    """Phase intervals and frame rate for one annotated case."""
    cid = case_dir_id(case_dir)
    fps = read_fps(case_dir / f"{cid}_video.csv")
    return read_phase_intervals(case_dir / f"{cid}_annotations_phases.csv", fps, pmap), fps


def _rows(path: Path) -> list[list[str]]:
    """Data rows of a CSV, header dropped, blank rows removed."""
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.reader(fh) if any(c.strip() for c in r)]
    return rows[1:]


def _case_number(name: str) -> int:
    digits = "".join(c for c in name if c.isdigit())
    return int(digits) if digits else 0
