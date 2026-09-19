"""Extract frames at configs/data.yaml fps and size; write manifest.csv with video_id, t_s, phase."""
import argparse

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--offline", action="store_true", help="assert no network access (always true in Environment B)")
    args = ap.parse_args()
    raise SystemExit("TODO: implement 01_extract_frames.py (see docs/ARCHITECTURE.md)")

if __name__ == "__main__":
    main()
