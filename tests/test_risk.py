from agaeye.decision import FrameDecision
from agaeye.risk import RiskConfig, RiskMonitor

NORMAL = FrameDecision(valid=True, posture="supine", airway="airway_visible")
PRONE = FrameDecision(valid=True, posture="prone", airway="airway_covered")
COVERED = FrameDecision(valid=True, posture="supine", airway="airway_covered")
NO_BABY = FrameDecision(valid=False, reason="no_baby")
COVERED_ONLY = FrameDecision(valid=True, reason="body_not_found", airway="airway_covered")


def run(monitor, frames, start=0.0, step=0.5):
    """(판단, 초) 목록을 step 간격 프레임으로 넣고 이벤트를 모아요."""
    events, t = [], start
    for decision, seconds in frames:
        for _ in range(int(round(seconds / step))):
            events += monitor.update(t, decision)
            t += step
    return events, t


def kinds(events):
    return [e.kind for e in events if e.kind != "status_changed"]


def test_prone_under_threshold_is_only_caution():
    m = RiskMonitor()
    events, _ = run(m, [(NORMAL, 1), (PRONE, 4.5)])
    assert kinds(events) == [] and m.status == "caution"


def test_prone_five_seconds_starts_risk():
    m = RiskMonitor()
    events, _ = run(m, [(NORMAL, 1), (PRONE, 6)])
    started = [e for e in events if e.kind == "risk_started"]
    assert len(started) == 1 and started[0].risk_type == "prone"
    assert started[0].confirmed_at - started[0].started_at >= 5.0
    assert m.status == "risk"


def test_face_covered_starts_risk():
    m = RiskMonitor()
    events, _ = run(m, [(COVERED, 6)])
    assert [e.risk_type for e in events if e.kind == "risk_started"] == ["face_covered"]


def test_risk_ends_only_after_three_seconds_normal():
    m = RiskMonitor()
    events, t = run(m, [(PRONE, 6)])
    events, t = run(m, [(NORMAL, 2.5)], start=t)
    assert kinds(events) == [] and m.status == "risk"
    events, t = run(m, [(NORMAL, 1)], start=t)
    assert kinds(events) == ["risk_ended"] and m.status == "normal" and m.active is None


def test_short_normal_blip_resets_clear_timer():
    m = RiskMonitor()
    _, t = run(m, [(PRONE, 6)])
    events, t = run(m, [(NORMAL, 2), (PRONE, 0.5), (NORMAL, 2.5)], start=t)
    assert kinds(events) == []  # 정상이 3초 연속이 아니었음


def test_short_gap_keeps_timer():
    m = RiskMonitor(RiskConfig(gap_tolerance_s=1.0))
    events, _ = run(m, [(PRONE, 3), (NO_BABY, 0.5), (PRONE, 2.5)])
    assert kinds(events) == ["risk_started"]


def test_long_gap_resets_timer():
    m = RiskMonitor(RiskConfig(gap_tolerance_s=1.0))
    events, _ = run(m, [(PRONE, 3), (NO_BABY, 2), (PRONE, 3)])
    assert kinds(events) == [] and m.status == "caution"


def test_strict_mode_resets_on_any_gap():
    m = RiskMonitor(RiskConfig(gap_tolerance_s=0.0))
    events, _ = run(m, [(PRONE, 3), (NO_BABY, 0.5), (PRONE, 3)])
    assert kinds(events) == []


def test_risk_survives_unavailable_until_normal_confirmed():
    m = RiskMonitor()
    _, t = run(m, [(PRONE, 6)])
    events, t = run(m, [(NO_BABY, 5)], start=t)
    assert kinds(events) == [] and m.status == "risk" and m.active is not None
    events, t = run(m, [(NORMAL, 3.5)], start=t)
    assert kinds(events) == ["risk_ended"]


def test_unavailable_status_when_no_baby():
    m = RiskMonitor()
    run(m, [(NORMAL, 1), (NO_BABY, 2)])
    assert m.status == "unavailable" and m.reason == "no_baby"


def test_brief_miss_keeps_timer():
    m = RiskMonitor(RiskConfig(miss_tolerance_s=1.0))
    events, _ = run(m, [(PRONE, 3), (NORMAL, 0.5), (PRONE, 2.5)])
    started = [e for e in events if e.kind == "risk_started"]
    assert len(started) == 1 and started[0].started_at == 0.0


def test_brief_miss_keeps_caution_status():
    m = RiskMonitor(RiskConfig(miss_tolerance_s=1.0))
    run(m, [(PRONE, 2), (NORMAL, 0.5)])
    assert m.status == "caution"


def test_long_miss_resets_timer():
    m = RiskMonitor(RiskConfig(miss_tolerance_s=1.0))
    events, _ = run(m, [(PRONE, 3), (NORMAL, 1.5), (PRONE, 3)])
    assert kinds(events) == []


def test_strict_miss_tolerance_resets_on_one_frame():
    m = RiskMonitor(RiskConfig(miss_tolerance_s=0.0))
    events, _ = run(m, [(PRONE, 3), (NORMAL, 0.5), (PRONE, 2.5)])
    assert kinds(events) == []


def test_confirmation_happens_on_a_frame_with_the_condition():
    m = RiskMonitor(RiskConfig(miss_tolerance_s=1.0))
    events, _ = run(m, [(PRONE, 5), (NORMAL, 0.5)])  # 5초가 되는 프레임에 안 잡힘
    assert kinds(events) == []
    events, _ = run(m, [(PRONE, 0.5)], start=5.5)
    assert kinds(events) == ["risk_started"]


def test_covered_face_without_body_starts_face_risk():
    m = RiskMonitor()
    events, _ = run(m, [(COVERED_ONLY, 6)])
    assert [e.risk_type for e in events if e.kind == "risk_started"] == ["face_covered"]


def test_threshold_change_applies():
    m = RiskMonitor(RiskConfig(risk_threshold_s=10))
    events, _ = run(m, [(PRONE, 6)])
    assert kinds(events) == []
    m.config.risk_threshold_s = 5
    events, _ = run(m, [(PRONE, 0.5)], start=6)
    assert kinds(events) == ["risk_started"]
