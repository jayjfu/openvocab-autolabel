import argparse
import shutil
from collections import Counter
from pathlib import Path

from utils.bdd100k import CLASSES, SPLITS
from utils.data_files import (load_split, load_teacher_predictions, read_class_thresholds, to_yolo_line,
                              write_data_config, write_lines, write_settings)
from utils.paths import ROOT
from utils.pseudo_labels import make_pseudo_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(ROOT / "data/bdd100k"))
    parser.add_argument("--label_threshold", type=float, default=0.35)
    parser.add_argument("--class_thresholds", help="yaml from calibrate_thresholds.py; overrides --label_threshold")
    parser.add_argument("--min_box_side", type=float, default=4)
    parser.add_argument("--max_area_fraction", type=float, default=0.6)
    parser.add_argument("--keep_ego_vehicle", action="store_true")
    parser.add_argument("--class_nms_iou", type=float, default=0.5)
    parser.add_argument("--cross_class_nms_iou", type=float, default=0.8)
    args = parser.parse_args()
    data_dir = Path(args.data_dir).resolve()
    filter_settings = dict(min_box_side=args.min_box_side, max_area_fraction=args.max_area_fraction,
                           drop_ego_vehicle=not args.keep_ego_vehicle, class_nms_iou=args.class_nms_iou,
                           cross_class_nms_iou=args.cross_class_nms_iou)
    label_threshold = args.label_threshold
    if args.class_thresholds:
        class_thresholds = read_class_thresholds(Path(args.class_thresholds))["thresholds"]
        label_threshold = [class_thresholds[class_name] for class_name in CLASSES]

    for split in SPLITS:
        label_dir = data_dir / "labels" / split
        shutil.rmtree(label_dir, ignore_errors=True)
        label_dir.mkdir(parents=True)
        (data_dir / "labels" / f"{split}.cache").unlink(missing_ok=True)

        image_paths, sizes, _ = load_split(data_dir, split)
        kept_per_class = Counter()
        for image_path, size, predictions in zip(image_paths, sizes, load_teacher_predictions(data_dir, split, image_paths)):
            pseudo_labels = make_pseudo_labels(predictions, *size, label_threshold, **filter_settings)
            class_ids, boxes = pseudo_labels["labels"].tolist(), pseudo_labels["boxes"].tolist()
            kept_per_class.update(CLASSES[class_id] for class_id in class_ids)
            write_lines(label_dir / f"{image_path.stem}.txt",
                        [to_yolo_line(class_id, box, *size) for class_id, box in zip(class_ids, boxes)])
        print(f"{split} pseudo boxes per class:", dict(kept_per_class.most_common()))

    write_data_config(data_dir, CLASSES)
    write_settings(data_dir, {"label_threshold": label_threshold, **filter_settings})


if __name__ == "__main__":
    main()