"""Prolonged-phaco marker: running phaco-phase duration against an empirical distribution."""
import numpy as np

class PhacoDurationRule:
    def __init__(self, durations_s, percentiles=(85, 95)):
        self.cuts = {p: float(np.percentile(durations_s, p)) for p in percentiles}
    def level(self, running_s: float):
        return sum(running_s >= c for c in self.cuts.values())   # 0, 1 or 2
