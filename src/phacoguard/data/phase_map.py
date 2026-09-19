"""Map dataset-specific phase names to PhacoGuard's seven phases."""
import yaml
PHASES = ["incision", "capsulorhexis", "hydrodissection", "phaco", "cortex_removal", "iol_insertion", "idle"]

def load_map(path="configs/phase_map.yaml"):
    return yaml.safe_load(open(path))

def to_phacoguard(dataset: str, label: str, pmap: dict) -> str:
    table = pmap.get(dataset) or {}
    out = table.get(label)
    if out is None:
        raise KeyError(f"Unmapped phase '{label}' for dataset '{dataset}'; add it to configs/phase_map.yaml")
    assert out in PHASES, out
    return out
