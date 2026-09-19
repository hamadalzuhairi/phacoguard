"""Fit pupil-constriction classifier, radial-fold clip classifier, phaco-duration percentiles; fit fusion + calibration on val."""
import argparse

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--offline", action="store_true", help="assert no network access (always true in Environment B)")
    args = ap.parse_args()
    raise SystemExit("TODO: implement 12_train_detectors.py (see docs/ARCHITECTURE.md)")

if __name__ == "__main__":
    main()
