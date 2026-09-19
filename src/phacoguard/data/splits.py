"""Video-level splits. Frames from one video never cross splits."""
import random

def video_split(video_ids, ratios=(0.7, 0.15, 0.15), seed=17):
    ids = sorted(set(video_ids)); rng = random.Random(seed); rng.shuffle(ids)
    n = len(ids); a = int(n * ratios[0]); b = a + int(n * ratios[1])
    return {"train": ids[:a], "val": ids[a:b], "test": ids[b:]}
