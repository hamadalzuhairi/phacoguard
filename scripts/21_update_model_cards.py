"""Fill docs/model_cards/*.md results rows from outputs/eval/*.json."""
import argparse

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--offline", action="store_true", help="assert no network access (always true in Environment B)")
    args = ap.parse_args()
    raise SystemExit("TODO: implement 21_update_model_cards.py (see docs/ARCHITECTURE.md)")

if __name__ == "__main__":
    main()
