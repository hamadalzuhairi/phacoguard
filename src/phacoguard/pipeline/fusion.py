"""Risk fusion with rate limiting. Output is an awareness state, never an instruction."""
STATES = ["routine", "rising", "high"]

class RiskFusion:
    def __init__(self, rate_limit_s=30.0):
        self.rate_limit_s = rate_limit_s; self.last = {}
    def allow(self, detector: str, t: float) -> bool:
        ok = t - self.last.get(detector, -1e9) >= self.rate_limit_s
        if ok: self.last[detector] = t
        return ok
    @staticmethod
    def state(flags: dict) -> str:
        n = int(flags.get("pupil_constriction", 0)) + int(flags.get("radial_folds", 0))
        phaco = int(flags.get("prolonged_phaco", 0))
        if n >= 1 and phaco >= 1: return "high"
        if n >= 1 or phaco >= 2: return "rising"
        return "routine"
