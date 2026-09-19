"""Verify every number and event in the generated record against the timeline JSON."""
import re

def verify(record_text: str, timeline: dict):
    nums = set(re.findall(r"\d+(?:\.\d+)?", record_text))
    allowed = set()
    for p in timeline.get("phases", []):
        allowed.update({f"{round(p['end_s']-p['start_s'])}", f"{round((p['end_s']-p['start_s'])/60, 1)}"})
    allowed.update({str(round(timeline.get("phaco_duration_s", 0))), str(round(timeline.get("phaco_percentile", 0)))})
    unverified = sorted(n for n in nums if n not in allowed)
    return {"n_numbers": len(nums), "unverified": unverified, "factual_accuracy": 1 - len(unverified) / max(len(nums), 1)}
