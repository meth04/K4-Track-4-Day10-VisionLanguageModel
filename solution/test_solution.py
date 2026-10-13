"""Contract checks for target geometry, supervision, and the bonus reward."""
import json
import pytest
from solution.data import answer, target_objects
from solution.reward import iou_reward

GT = [{"label": "pothole", "bbox": (100, 100, 300, 300)}]
BOX = {"label": "pothole", "bbox_2d": [100, 100, 300, 300]}


@pytest.mark.parametrize("raw,gt,expected", [
    (json.dumps([BOX]), GT, 1.0),
    ("broken json", GT, 0.0),
    (json.dumps([{**BOX, "label": "crack"}]), GT, 0.2),
    (json.dumps([BOX, BOX]), GT, 0.6),
    ("[]", [], 1.0),
    ("[]", GT, 0.2),
    (json.dumps([BOX]), [], 0.2),
    (json.dumps([{**BOX, "bbox_2d": [0, 0, 1000, 1000]}]), GT, 0.2),
])
def test_reward_contract(raw, gt, expected):
    assert iou_reward(raw, gt) == pytest.approx(expected)


def test_round_trip_coordinate_scales():
    entry = {"image_id": "test", "width": 640, "height": 360, "objects": [{"label": "pothole", "yolo": [.5, .5, .25, .5]}]}
    assert target_objects(entry)[0]["bbox_2d"] == [375, 250, 625, 750]
    assert target_objects(entry, "pixel")[0]["bbox_2d"] == [240, 90, 400, 270]
    assert json.loads(answer({**entry, "objects": []})) == []


def test_duplicate_prediction_cannot_reuse_ground_truth():
    assert iou_reward(json.dumps([BOX]*10), GT) == pytest.approx(.28)
