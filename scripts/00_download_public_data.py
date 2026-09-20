"""Download public datasets to --root. PUBLIC DATA ONLY.

Currently implements Cataract-1K (CC BY 4.0) from Synapse. Network access lives
here in `scripts/`, never in `src/phacoguard` (CLAUDE.md rule 2).

Synapse allows anonymous metadata reads but not anonymous downloads, so a
personal access token is required. Create one at
synapse.org -> Account Settings -> Personal Access Tokens (scope: Download) and
export it; the token is never written to the repository.

    export SYNAPSE_AUTH_TOKEN=...            # bash
    $env:SYNAPSE_AUTH_TOKEN = "..."          # PowerShell

    python scripts/00_download_public_data.py --subset phase --accept-licence
    python scripts/00_download_public_data.py --subset phase --videos case_5015 case_5353
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

SYNAPSE = "https://repo-prod.prod.sagebase.org/repo/v1"

CATARACT1K = {
    "project": "syn52540135",
    "phase_annotations": "syn53395153",
    "phase_videos": "syn53395325",
    "pupil_reaction": "syn53395402",
    "lens_irregularity": "syn53395131",
}
LICENCE = "CC BY 4.0"
CITATION = (
    "Ghamsarian, N. et al. Cataract-1K: annotated dataset for cataract surgery "
    "video analysis. Scientific Data 11, 373 (2024). Synapse syn52540135."
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--root", default="data/cataract1k", help="destination directory")
    ap.add_argument("--subset", choices=["phase", "pupil_reaction", "lens_irregularity"],
                    default="phase")
    ap.add_argument("--videos", nargs="*", default=None,
                    help="case ids to fetch videos for (default: annotations only)")
    ap.add_argument("--all-videos", action="store_true", help="fetch every video in the subset")
    ap.add_argument("--accept-licence", action="store_true",
                    help="confirm the CC BY 4.0 terms and attribution requirement")
    ap.add_argument("--offline", action="store_true",
                    help="assert no network access (always true in Environment B)")
    args = ap.parse_args()

    if args.offline:
        raise SystemExit("--offline: this script downloads data and must not run in Environment B")

    print(f"Dataset : Cataract-1K\nLicence : {LICENCE}\nCite    : {CITATION}\n")
    if not args.accept_licence:
        raise SystemExit(
            "Re-run with --accept-licence to confirm the CC BY 4.0 attribution requirement.\n"
            "Attribution must appear in the demo video and in docs/DATA_AND_LICENSES.md."
        )

    token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    if not token:
        raise SystemExit(
            "SYNAPSE_AUTH_TOKEN is not set. Create a personal access token at\n"
            "  https://www.synapse.org/#!PersonalAccessTokens:0\n"
            "with the Download scope, then export it as SYNAPSE_AUTH_TOKEN."
        )

    root = Path(args.root)
    if args.subset == "phase":
        n = _phase_subset(root, token, args.videos, args.all_videos)
    else:
        n = _flat_subset(root / args.subset, token, CATARACT1K[args.subset],
                         args.videos, args.all_videos)
    print(f"\nDone. {n} file(s) under {root.resolve()}")
    _write_attribution(root)


def _phase_subset(root: Path, token: str, videos: list[str] | None, all_videos: bool) -> int:
    n = 0
    ann_root = root / "annotations"
    for case in _children(CATARACT1K["phase_annotations"], token):
        if not case["type"].endswith("Folder"):
            continue
        dest = ann_root / case["name"]
        for f in _children(case["id"], token):
            n += _download(f["id"], dest / f["name"], token)
    wanted = set(videos or [])
    if wanted or all_videos:
        for f in _children(CATARACT1K["phase_videos"], token):
            stem = f["name"].rsplit(".", 1)[0]
            if all_videos or stem in wanted:
                n += _download(f["id"], root / "videos" / f["name"], token)
    return n


def _flat_subset(dest: Path, token: str, parent: str, videos: list[str] | None,
                 all_videos: bool) -> int:
    wanted = set(videos or [])
    n = 0
    for f in _children(parent, token):
        stem = f["name"].rsplit(".", 1)[0]
        if all_videos or not wanted or stem in wanted:
            n += _download(f["id"], dest / f["name"], token)
    return n


def _children(parent_id: str, token: str) -> list[dict]:
    out, tok = [], None
    while True:
        body = {"parentId": parent_id, "includeTypes": ["folder", "file"],
                "sortBy": "NAME", "sortDirection": "ASC"}
        if tok:
            body["nextPageToken"] = tok
        data = _api("/entity/children", token, body)
        out += data.get("page", [])
        tok = data.get("nextPageToken")
        if not tok:
            return out


def _download(syn_id: str, dest: Path, token: str) -> int:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip  {dest.name} (present)")
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(
            f"{SYNAPSE}/entity/{syn_id}/file?redirect=false",
            headers={"Authorization": f"Bearer {token}"},
        )
        url = urllib.request.urlopen(req, timeout=60).read().decode()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as fh:
            while chunk := r.read(1 << 20):
                fh.write(chunk)
        tmp.replace(dest)
        print(f"  got   {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return 1
    except urllib.error.HTTPError as e:
        print(f"  FAIL  {dest.name}: HTTP {e.code} {e.reason}", file=sys.stderr)
        return 0


def _api(path: str, token: str, body: dict) -> dict:
    req = urllib.request.Request(
        SYNAPSE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    return json.load(urllib.request.urlopen(req, timeout=60))


def _write_attribution(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "ATTRIBUTION.txt").write_text(
        f"Cataract-1K\nLicence: {LICENCE}\n{CITATION}\n\n"
        "Attribution must be shown wherever this footage appears, including the demo video.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
