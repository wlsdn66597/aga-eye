"""평가의 판정 비교(score)와 기준값 고르기(pick)를 가짜 박스로 확인해요."""
import importlib.util
import sys
from pathlib import Path

from agaeye.decision import Detection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gt(body_cls, face_cls):
    return [Detection(body_cls, 0.5, (0, 0, 1, 1)), Detection(face_cls, 0.01, (0.4, 0.1, 0.6, 0.3))]


def body(cls, conf):
    return Detection(cls, conf, (0, 0, 100, 100))


def face(cls, conf):
    return Detection(cls, conf, (40, 10, 60, 30))


COLLECTED = {
    "weights": "w", "data": "d", "split": "test", "imgsz": 416, "backend": "test",
    "images": ["a.jpg", "b.jpg", "c.jpg"], "sources": ["s1", "s1", "s2"],
    "gts": [gt("prone", "airway_covered"),       # 엎드림
            gt("supine", "airway_covered"),      # 얼굴 가림
            gt("supine", "airway_visible")],     # 정상
    "preds": [[body("prone", 0.6)],
              [face("airway_covered", 0.3)],     # 몸 박스 못 찾고 가림 얼굴만 (신뢰도 0.3)
              [body("supine", 0.9), face("airway_visible", 0.8)]],
}


def test_strict_setting_misses_covered_face_without_body():
    m = load_script("evaluate").score(COLLECTED, conf_body=0.5, conf_airway=0.4, covered_without_body=False)
    assert (m["risk"]["tp"], m["risk"]["fn"], m["risk"]["fp"]) == (1, 1, 0)
    assert m["risk_by_type"]["face_covered"]["recall_any_risk"] == 0.0


def test_looser_setting_catches_it():
    m = load_script("evaluate").score(COLLECTED, conf_body=0.5, conf_airway=0.25, covered_without_body=True)
    assert (m["risk"]["tp"], m["risk"]["fn"], m["risk"]["fp"]) == (2, 0, 0)
    assert m["risk"]["recall"] == 1.0 and m["by_source"]["s2"]["risk_accuracy"] == 1.0


def test_pick_prefers_precision_among_rows_meeting_target():
    pick = load_script("sweep_thresholds").pick
    rows = [{"risk_recall": 0.95, "risk_precision": 0.70}, {"risk_recall": 0.91, "risk_precision": 0.85},
            {"risk_recall": 0.80, "risk_precision": 0.99}]
    assert pick(rows, 0.9) == rows[1]
    assert pick(rows, 0.99) == rows[0]  # 목표를 넘는 게 없으면 재현율 최고
