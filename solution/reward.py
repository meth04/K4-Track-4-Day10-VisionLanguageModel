"""Verifiable IoU reward; pure Python and testable without a GPU."""
from scripts.score import iou_xyxy, parse_output


def iou_reward(completion, gt_boxes):
    pred = parse_output(completion)
    if pred is None:
        return 0.0
    if not pred and not gt_boxes:
        return 1.0
    pairs = sorted(((iou_xyxy(g["bbox"], p["bbox"]), i, j) for i, g in enumerate(gt_boxes) for j, p in enumerate(pred) if g["label"] == p["label"]), reverse=True)
    used_gt, used_pred, total = set(), set(), 0.0
    for overlap, i, j in pairs:
        if overlap < 0.5:
            break
        if i not in used_gt and j not in used_pred:
            used_gt.add(i)
            used_pred.add(j)
            total += overlap
    return 0.2 + 0.8 * total / max(len(gt_boxes), len(pred), 1)


def grounding_reward(completions, gt_boxes, **kwargs):
    texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
    return [iou_reward(text, gt) for text, gt in zip(texts, gt_boxes, strict=True)]
