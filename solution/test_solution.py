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


@pytest.mark.parametrize('coord', ['norm1000','pixel'])
@pytest.mark.parametrize('center', [.0,.5,1.0])
def test_subpixel_boxes_remain_valid_at_boundaries(coord,center):
    entry={'image_id':'thin','width':640,'height':360,'objects':[{'label':'crack','yolo':[center,center,.004,.0001]}]}
    bb=target_objects(entry,coord)[0]['bbox_2d']
    sx,sy=(1000,1000) if coord=='norm1000' else (640,360)
    assert 0<=bb[0]<bb[2]<=sx
    assert 0<=bb[1]<bb[3]<=sy


def test_report_error_counts_agree_with_official_scorer():
    from scripts.score import evaluate
    from solution.reporting import error_breakdown
    entries=[
        {"image_id":"a","width":640,"height":360,"objects":[{"label":"pothole","yolo":[.2,.2,.2,.2]},{"label":"crack","yolo":[.5,.5,.2,.2]}]},
        {"image_id":"b","width":640,"height":360,"objects":[{"label":"pothole","yolo":[.2,.2,.2,.2]}]},
    ]
    predictions={"a":json.dumps([BOX,BOX,{"label":"manhole","bbox_2d":[400,400,600,600]}]),"b":"broken JSON"}
    counts=error_breakdown(entries,predictions)['counts']
    official=evaluate(entries,predictions,n_boot=40)['micro']
    assert counts['bỏ sót GT']==official['fn']==2
    assert sum(counts[k] for k in ('box trùng','sai lớp với IoU ≥ 0.5','sai định vị, IoU cùng lớp 0.1–0.5','FP có IoU cùng lớp < 0.1'))==official['fp']==2
    assert counts['box trùng']==counts['sai lớp với IoU ≥ 0.5']==counts['JSON hỏng']==1
