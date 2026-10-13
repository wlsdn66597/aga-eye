from agaeye.decision import Detection, decide


def det(cls, conf, box=(0, 0, 100, 100)):
    return Detection(cls, conf, box)


def test_no_body_is_unavailable():
    d = decide([det("airway_visible", 0.9)])
    assert not d.valid and d.reason == "no_baby" and d.risk_type is None


def test_low_confidence_body_is_ignored():
    assert not decide([det("supine", 0.3)], conf_body=0.5).valid


def test_highest_confidence_body_wins():
    d = decide([det("supine", 0.6), det("prone", 0.9)])
    assert d.posture == "prone" and d.risk_type == "prone"


def test_face_covered_when_not_prone():
    d = decide([det("supine", 0.9), det("airway_covered", 0.8, (40, 10, 60, 30))])
    assert d.airway == "airway_covered" and d.risk_type == "face_covered"


def test_prone_takes_priority_over_face_covered():
    d = decide([det("prone", 0.9), det("airway_covered", 0.8, (40, 10, 60, 30))])
    assert d.risk_type == "prone"


def test_face_box_outside_body_is_ignored():
    d = decide([det("supine", 0.9, (0, 0, 100, 100)), det("airway_covered", 0.9, (300, 300, 340, 340))])
    assert d.airway is None and d.risk_type is None


def test_missing_face_box_is_not_treated_as_covered():
    d = decide([det("side", 0.9)])
    assert d.valid and d.airway is None and d.risk_type is None


def test_covered_face_without_body_is_ignored_by_default():
    assert not decide([det("airway_covered", 0.9)]).valid


def test_covered_face_without_body_counts_when_enabled():
    d = decide([det("airway_covered", 0.9)], covered_without_body=True)
    assert d.valid and d.posture is None and d.risk_type == "face_covered" and d.reason == "body_not_found"


def test_visible_face_without_body_stays_unavailable():
    assert not decide([det("airway_visible", 0.9)], covered_without_body=True).valid
