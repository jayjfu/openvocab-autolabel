import argparse
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

from utils.bdd100k import CLASSES
from utils.formats import read_scored_boxes, read_yolo_boxes
from utils.paths import ROOT
from utils.metrics import mean_average_precision, precision_recall


def load_split(data_dir, split):
    image_paths = sorted((data_dir / "images" / split).glob("*.jpg"))
    targets = [read_yolo_boxes(data_dir / "labels_human" / split / f"{path.stem}.txt", *Image.open(path).size)
               for path in image_paths]
    return image_paths, targets


def load_teacher_predictions(data_dir, split, image_paths, min_score=0.0):
    return [read_scored_boxes(data_dir / "predictions_teacher" / split / f"{path.stem}.txt", min_score)
            for path in image_paths]


def predict_student(weights, image_paths, image_size):
    results = YOLO(weights).predict(source=[str(path) for path in image_paths], conf=0.001, imgsz=image_size,
                                    stream=True, verbose=False)
    return [{"boxes": result.boxes.xyxy.cpu(), "scores": result.boxes.conf.cpu(), "labels": result.boxes.cls.cpu().long()}
            for result in results]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(ROOT / "data/bdd100k"))
    parser.add_argument("--student_weights", default=str(ROOT / "src/openvocab-autolabel/ckpts/student/weights/best.pt"))
    parser.add_argument("--image_size", type=int, default=640)
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    print("pseudo-label quality on train (IoU 0.5)")
    train_paths, train_targets = load_split(data_dir, "train")
    for threshold in [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        pseudo_labels = load_teacher_predictions(data_dir, "train", train_paths, threshold)
        precision, recall = precision_recall(pseudo_labels, train_targets)
        print(f"  threshold {threshold:.2f}: precision {precision:.3f}  recall {recall:.3f}")

    test_paths, test_targets = load_split(data_dir, "test")
    models = {"teacher": load_teacher_predictions(data_dir, "test", test_paths),
              "student": predict_student(args.student_weights, test_paths, args.image_size)}
    for model_name, predictions in models.items():
        map_all, map_50, per_class = mean_average_precision(predictions, test_targets, CLASSES)
        print(f"{model_name} on test: mAP50-95 {map_all:.3f}  mAP50 {map_50:.3f}")
        print(f"  per-class mAP50-95: {per_class}")


if __name__ == "__main__":
    main()