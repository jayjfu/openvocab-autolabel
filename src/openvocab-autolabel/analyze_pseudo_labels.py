import argparse
from pathlib import Path

from utils.bdd100k import CLASSES
from utils.data_files import load_split, load_teacher_predictions, read_class_thresholds
from utils.metrics import mean_average_precision, precision_recall
from utils.paths import ROOT
from utils.pseudo_labels import filter_predictions, make_pseudo_labels

FILTER_STEPS = [
    ("raw", dict(min_box_side=0, max_area_fraction=1.0, drop_ego_vehicle=False, class_nms_iou=None,
                 cross_class_nms_iou=None)),
    ("+ box size", dict(drop_ego_vehicle=False, class_nms_iou=None, cross_class_nms_iou=None)),
    ("+ ego vehicle", dict(class_nms_iou=None, cross_class_nms_iou=None)),
    ("+ class NMS", dict(cross_class_nms_iou=None)),
    ("+ cross-class NMS", dict()),
]
THRESHOLDS = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(ROOT / "data/bdd100k"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--label_threshold", type=float, default=0.35)
    parser.add_argument("--class_thresholds", default=str(ROOT / "output/calibration/class_thresholds.yaml"))
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    image_paths, sizes, targets = load_split(data_dir, args.split)
    predictions = load_teacher_predictions(data_dir, args.split, image_paths)

    print(f"filter steps at threshold {args.label_threshold} ({args.split}, {len(image_paths)} images)")
    for step_name, filter_settings in FILTER_STEPS:
        filtered = [filter_predictions(prediction, *size, **filter_settings) for prediction, size in zip(predictions, sizes)]
        pseudo_labels = [make_pseudo_labels(prediction, *size, args.label_threshold, **filter_settings)
                         for prediction, size in zip(predictions, sizes)]
        precision, recall = precision_recall(pseudo_labels, targets)
        _, map_50, _ = mean_average_precision(filtered, targets, CLASSES)
        num_boxes = sum(len(labels["labels"]) for labels in pseudo_labels)
        print(f"  {step_name:<20} boxes {num_boxes:>7}  precision {precision:.3f}  recall {recall:.3f}  teacher mAP50 {map_50:.3f}")

    print("thresholds with all filters")
    for threshold in THRESHOLDS:
        pseudo_labels = [make_pseudo_labels(prediction, *size, threshold) for prediction, size in zip(predictions, sizes)]
        precision, recall = precision_recall(pseudo_labels, targets)
        print(f"  threshold {threshold:.2f}: precision {precision:.3f}  recall {recall:.3f}")

    if Path(args.class_thresholds).exists():
        calibration = read_class_thresholds(Path(args.class_thresholds))
        held_out = slice(calibration["num_calibration_images"], None) if args.split == "train" else slice(None)
        class_thresholds = [calibration["thresholds"][class_name] for class_name in CLASSES]
        print(f"global vs per-class thresholds ({len(image_paths[held_out])} images not used for calibration)")
        for name, label_threshold in [(f"global {args.label_threshold}", args.label_threshold), ("per-class", class_thresholds)]:
            pseudo_labels = [make_pseudo_labels(prediction, *size, label_threshold)
                             for prediction, size in zip(predictions[held_out], sizes[held_out])]
            precision, recall = precision_recall(pseudo_labels, targets[held_out])
            f_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            print(f"  {name:<12} precision {precision:.3f}  recall {recall:.3f}  F1 {f_score:.3f}")


if __name__ == "__main__":
    main()