"""Roboflow 형식의 가짜 데이터셋으로 prepare_dataset.py를 확인해요."""
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW_NAMES = ["Active-Parent", "Airway-Covered", "Airway-Visible", "Baby-Prone",
             "Baby-Sideways", "Baby-Supine", "Toy-Beside"]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_raw(raw: Path):
    files = {
        "train": {"1190_png.rf.a.jpg": "3 0.5 0.5 0.6 0.6\n1 0.5 0.3 0.1 0.1\n6 0.1 0.1 0.05 0.05",
                  "Sideways_1_jpg.rf.b.jpg": "4 0.5 0.5 0.5 0.5\n2 0.5 0.3 0.1 0.1",
                  "IMG_8761_JPG.rf.c.jpg": "5 0.5 0.5 0.5 0.5\n1 0.5 0.3 0.1 0.1"},
        "valid": {"Supine_3_jpg.rf.d.jpg": "5 0.5 0.5 0.5 0.5\n2 0.5 0.3 0.1 0.1"},
        "test": {"IMG_8762_JPG.rf.e.jpg": "0 0.5 0.5 0.2 0.2"},
    }
    for split, items in files.items():
        (raw / split / "images").mkdir(parents=True)
        (raw / split / "labels").mkdir(parents=True)
        for name, label in items.items():
            # cv2.imwrite는 윈도우 한글 경로에서 실패해서 imencode + tofile로 저장
            cv2.imencode(".jpg", np.zeros((32, 32, 3), np.uint8))[1].tofile(str(raw / split / "images" / name))
            (raw / split / "labels" / (Path(name).stem + ".txt")).write_text(label)
    (raw / "data.yaml").write_text(yaml.safe_dump({"names": RAW_NAMES, "nc": len(RAW_NAMES)}))


def test_prepare_remaps_and_splits(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    make_raw(raw)
    load_script("prepare_dataset").main(["--raw", str(raw), "--out", str(out)])

    # 클래스 바꾸기: Baby-Prone(3)→prone(2), Airway-Covered(1)→airway_covered(4), Toy-Beside 버림
    lines = (out / "labels" / "1190_png.rf.a.txt").read_text().split("\n")
    assert [line.split()[0] for line in lines if line] == ["2", "4"]
    # 보호자 손만 있던 사진은 빈 라벨(배경)
    assert (out / "labels" / "IMG_8762_JPG.rf.e.txt").read_text() == ""

    original = yaml.safe_load((out / "data_original.yaml").read_text(encoding="utf-8"))
    assert original["names"] == ["supine", "side", "prone", "airway_visible", "airway_covered"]
    assert len(Path(original["train"]).read_text().splitlines()) == 3

    holdout = yaml.safe_load((out / "data_holdout_doll_blanket.yaml").read_text(encoding="utf-8"))
    test_files = [Path(p).name for p in Path(holdout["test"]).read_text().splitlines()]
    train_files = [Path(p).name for p in Path(holdout["train"]).read_text().splitlines()]
    assert sorted(test_files) == ["IMG_8761_JPG.rf.c.jpg", "IMG_8762_JPG.rf.e.jpg"]
    assert not set(test_files) & set(train_files)


def test_other_label_scheme_dataset_with_split_all(tmp_path):
    """라벨 방식이 다른 데이터셋을 class_map으로 정리하고 전체 사진(split=all)으로 평가해요."""
    raw, out = tmp_path / "raw", tmp_path / "out"
    config = tmp_path / "dataset_other.yaml"
    config.write_text(yaml.safe_dump({
        "raw_dir": str(raw), "out_dir": str(out),
        "classes": ["supine", "side", "prone", "airway_visible", "airway_covered"],
        "class_map": {"normal": "supine", "overturn": "prone", "covered_face": "airway_covered"},
        "source_groups": [], "default_group": "other", "holdout_groups": []}), encoding="utf-8")
    names = ["covered_face", "normal", "overturn"]
    for split, items in {"train": {"IMG_1.jpg": "1 0.5 0.5 0.4 0.8", "IMG_2.jpg": "0 0.5 0.5 0.4 0.8"},
                         "test": {"IMG_3.jpg": "2 0.5 0.5 0.4 0.8"}}.items():
        (raw / split / "images").mkdir(parents=True)
        (raw / split / "labels").mkdir(parents=True)
        for name, label in items.items():
            cv2.imencode(".jpg", np.zeros((32, 32, 3), np.uint8))[1].tofile(str(raw / split / "images" / name))
            (raw / split / "labels" / (Path(name).stem + ".txt")).write_text(label)
    (raw / "data.yaml").write_text(yaml.safe_dump({"names": names, "nc": 3}))

    load_script("prepare_dataset").main(["--config", str(config)])
    ev = load_script("evaluate")
    data = out / "data_original.yaml"
    imgs = ev.split_images(data, "all")
    assert sorted(p.name for p in imgs) == ["IMG_1.jpg", "IMG_2.jpg", "IMG_3.jpg"]

    names_out = ev.data_names(data)
    gts = {p.name: ev.decide(ev.ground_truth(p, names_out), 0.0, 0.0, covered_without_body=True) for p in imgs}
    assert gts["IMG_1.jpg"].risk_type is None              # normal → 위험 없음
    assert gts["IMG_2.jpg"].risk_type == "face_covered"    # covered_face → 얼굴 가림
    assert gts["IMG_3.jpg"].risk_type == "prone"           # overturn → 엎드림
