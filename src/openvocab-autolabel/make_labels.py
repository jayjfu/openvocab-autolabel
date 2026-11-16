import argparse
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

from utils.bdd100k import CLASSES, SPLITS
from utils.formats import read_scored_boxes, to_yolo_line, write_data_config
from utils.paths import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(ROOT / "data/bdd100k"))
    parser.add_argument("--label_threshold", type=float, default=0.35)
    args = parser.parse_args()
    data_dir = Path(args.data_dir).resolve()

    for split in SPLITS:
        label_dir = data_dir / "labels" / split
        shutil.rmtree(label_dir, ignore_errors=True)
        label_dir.mkdir(parents=True)
        (data_dir / "labels" / f"{split}.cache").unlink(missing_ok=True)

        kept_per_class = Counter()
        for prediction_path in sorted((data_dir / "predictions_teacher" / split).glob("*.txt")):
            width, height = Image.open(data_dir / "images" / split / f"{prediction_path.stem}.jpg").size
            predictions = read_scored_boxes(prediction_path, args.label_threshold)
            class_ids, boxes = predictions["labels"].tolist(), predictions["boxes"].tolist()
            kept_per_class.update(CLASSES[class_id] for class_id in class_ids)
            (label_dir / prediction_path.name).write_text(
                "\n".join(to_yolo_line(class_id, box, width, height) for class_id, box in zip(class_ids, boxes)))
        print(f"{split} pseudo boxes per class:", dict(kept_per_class.most_common()))

    write_data_config(data_dir, CLASSES)


if __name__ == "__main__":
    main()
