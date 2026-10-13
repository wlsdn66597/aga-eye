import numpy as np

from agaeye.detector_onnx import letterbox, nms, postprocess


def test_letterbox_keeps_ratio_and_pads():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    out, r, (left, top) = letterbox(img, 416)
    assert out.shape == (416, 416, 3)
    assert abs(r - 0.65) < 1e-6 and left == 0 and top == 52


def test_nms_removes_overlap_keeps_separate():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    keep = nms(boxes, np.array([0.9, 0.8, 0.7], dtype=np.float32), 0.5)
    assert keep.tolist() == [0, 2]


def test_postprocess_v8_output_maps_back_to_original():
    # 클래스 2개, 후보 3개: (cx, cy, w, h, score0, score1)
    cand = np.array([[208, 208, 100, 50, 0.9, 0.1],
                     [210, 210, 100, 50, 0.2, 0.8],    # 다른 클래스라 겹쳐도 남음
                     [100, 100, 20, 20, 0.05, 0.1]],   # 신뢰도 낮아 버림
                    dtype=np.float32)
    out = cand.T[None]  # (1, 6, 3)  ← 6행이라도 v8 형식으로 처리돼야 함
    r, pad = 0.65, (0, 52)
    xyxy, conf, cls = postprocess(out, 0.25, 0.7, r, pad, (480, 640))
    assert cls.tolist() == [0, 1]
    assert np.allclose(xyxy[0], [(158 - 0) / r, (183 - 52) / r, (258 - 0) / r, (233 - 52) / r], atol=1e-3)


def test_postprocess_end2end_output():
    out = np.zeros((1, 300, 6), dtype=np.float32)
    out[0, 0] = [10, 10, 50, 50, 0.9, 2]
    out[0, 1] = [0, 0, 5, 5, 0.1, 1]
    xyxy, conf, cls = postprocess(out, 0.25, 0.7, 1.0, (0, 0), (100, 100))
    assert cls.tolist() == [2] and np.allclose(xyxy[0], [10, 10, 50, 50])
