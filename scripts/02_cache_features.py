"""Run the frozen ResNet-50 once over all frames; save features per video (.npy)."""
import argparse

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--offline", action="store_true", help="assert no network access (always true in Environment B)")
    args = ap.parse_args()
    raise SystemExit("TODO: implement 02_cache_features.py (see docs/ARCHITECTURE.md)")

if __name__ == "__main__":
    main()
