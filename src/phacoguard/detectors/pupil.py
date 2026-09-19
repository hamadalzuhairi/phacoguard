"""Pupil-constriction marker from a normalised pupil-area time series.
Feature extraction is fixed; the decision threshold is learned on the Cataract-1K pupil-reaction subset."""
import numpy as np

def window_features(area: np.ndarray, fps: float, window_s: float = 30.0):
    n = int(window_s * fps)
    if len(area) < n: return None
    w = area[-n:]
    running_max = np.maximum.accumulate(area)[-1]
    rel_drop = (running_max - w[-1]) / max(running_max, 1e-6)
    slope = np.polyfit(np.arange(n), w, 1)[0]
    return np.array([rel_drop, slope, w.var()])
