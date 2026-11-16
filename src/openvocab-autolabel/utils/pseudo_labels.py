import torch
from torchvision.ops import batched_nms, nms

from utils.metrics import match_predictions


def select(predictions, keep):
    return {key: value[keep] for key, value in predictions.items()}


def filter_predictions(predictions, width, height, min_box_side=4, max_area_fraction=0.6, drop_ego_vehicle=True,
                       class_nms_iou=0.5, cross_class_nms_iou=0.8):
    boxes = predictions["boxes"]
    box_width, box_height = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    keep = (box_width >= min_box_side) & (box_height >= min_box_side)
    keep &= box_width * box_height <= max_area_fraction * width * height
    if drop_ego_vehicle:
        keep &= ~((boxes[:, 3] >= 0.97 * height) & (box_width >= 0.7 * width))
    predictions = select(predictions, keep)
    if class_nms_iou is not None:
        predictions = select(predictions, batched_nms(predictions["boxes"], predictions["scores"],
                                                      predictions["labels"], class_nms_iou))
    if cross_class_nms_iou is not None:
        predictions = select(predictions, nms(predictions["boxes"], predictions["scores"], cross_class_nms_iou))
    return predictions


def make_pseudo_labels(predictions, width, height, label_threshold, **filter_settings):
    filtered = filter_predictions(predictions, width, height, **filter_settings)
    threshold = torch.as_tensor(label_threshold)
    if threshold.ndim:
        threshold = threshold[filtered["labels"]]
    return select(filtered, filtered["scores"] >= threshold)


def calibrate_thresholds(filtered_predictions, targets, class_names, candidate_thresholds, beta=1.0, min_targets=10,
                         default_threshold=0.35):
    scores, labels, is_true, target_labels = match_predictions(filtered_predictions, targets)
    thresholds, report = {}, {}
    for class_id, class_name in enumerate(class_names):
        num_targets = int((target_labels == class_id).sum())
        if num_targets < min_targets:
            thresholds[class_name] = default_threshold
            report[class_name] = {"targets": num_targets, "threshold": default_threshold, "note": "too few targets"}
            continue
        class_scores, class_true = scores[labels == class_id], is_true[labels == class_id]
        best = None
        for threshold in candidate_thresholds:
            kept = class_scores >= threshold
            true_positives = int(class_true[kept].sum())
            precision = true_positives / max(int(kept.sum()), 1)
            recall = true_positives / num_targets
            f_score = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall) if precision + recall else 0.0
            if best is None or f_score >= best["f_score"]:
                best = {"targets": num_targets, "threshold": threshold, "precision": round(precision, 3),
                        "recall": round(recall, 3), "f_score": round(f_score, 3)}
        thresholds[class_name] = best["threshold"]
        report[class_name] = best
    return thresholds, report