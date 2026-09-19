from phacoguard.pipeline.fusion import RiskFusion
def test_states():
    assert RiskFusion.state({}) == "routine"
    assert RiskFusion.state({"pupil_constriction": 1}) == "rising"
    assert RiskFusion.state({"radial_folds": 1, "prolonged_phaco": 1}) == "high"
def test_rate_limit():
    f = RiskFusion(30); assert f.allow("x", 0) and not f.allow("x", 10) and f.allow("x", 31)
