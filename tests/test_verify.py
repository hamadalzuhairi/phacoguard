from phacoguard.report.verify import verify
def test_verify_catches_invented_numbers():
    tl = {"phases": [{"phase": "phaco", "start_s": 0, "end_s": 240}], "phaco_duration_s": 240, "phaco_percentile": 90}
    r = verify("Phaco lasted 240 s (90th percentile). Pupil area fell 55%.", tl)
    assert "55" in r["unverified"]
