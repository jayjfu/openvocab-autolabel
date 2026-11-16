import torch
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou


def mean_average_precision(predictions, targets, class_names):
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric.update(predictions, targets)
    result = metric.compute()
    per_class = {class_names[int(class_id)]: round(float(value), 3)
                 for class_id, value in zip(result["classes"], result["map_per_class"])}
    return float(result["map"]), float(result["map_50"]), per_class


def match_predictions(predictions, targets, iou_threshold=0.5):
    scores, labels, is_true, target_labels = [], [], [], []
    for prediction, target in zip(predictions, targets):
        prediction_true = torch.zeros(len(prediction["labels"]), dtype=torch.bool)
        if len(prediction["labels"]) and len(target["labels"]):
            iou = box_iou(prediction["boxes"], target["boxes"])
            iou[prediction["labels"][:, None] != target["labels"][None, :]] = 0
            target_matched = torch.zeros(len(target["labels"]), dtype=torch.bool)
            for prediction_index in prediction["scores"].argsort(descending=True):
                candidate_iou = iou[prediction_index].masked_fill(target_matched, 0)
                best_index = candidate_iou.argmax()
                if candidate_iou[best_index] >= iou_threshold:
                    target_matched[best_index] = True
                    prediction_true[prediction_index] = True
        scores.append(prediction["scores"])
        labels.append(prediction["labels"])
        is_true.append(prediction_true)
        target_labels.append(target["labels"])
    return torch.cat(scores), torch.cat(labels), torch.cat(is_true), torch.cat(target_labels)


def precision_recall(predictions, targets, iou_threshold=0.5):
    _, _, is_true, target_labels = match_predictions(predictions, targets, iou_threshold)
    true_positives = int(is_true.sum())
    return true_positives / max(len(is_true), 1), true_positives / max(len(target_labels), 1)