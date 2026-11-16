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


def precision_recall(predictions, targets, iou_threshold=0.5):
    true_positives = num_predicted = num_actual = 0
    for prediction, target in zip(predictions, targets):
        num_predicted += len(prediction["labels"])
        num_actual += len(target["labels"])
        if not len(prediction["labels"]) or not len(target["labels"]):
            continue
        iou = box_iou(prediction["boxes"], target["boxes"])
        iou[prediction["labels"][:, None] != target["labels"][None, :]] = 0
        matched = torch.zeros(len(target["labels"]), dtype=torch.bool)
        for prediction_index in prediction["scores"].argsort(descending=True):
            candidate_iou = iou[prediction_index].masked_fill(matched, 0)
            best_index = candidate_iou.argmax()
            if candidate_iou[best_index] >= iou_threshold:
                matched[best_index] = True
                true_positives += 1
    return true_positives / max(num_predicted, 1), true_positives / max(num_actual, 1)
