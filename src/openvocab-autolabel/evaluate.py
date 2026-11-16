import argparse
from pathlib import Path

from ultralytics import YOLO

from utils.bdd100k import CLASSES
from utils.data_files import load_split, load_teacher_predictions, read_settings
from utils.metrics import mean_average_precision
from utils.paths import ROOT
from utils.pseudo_labels import filter_predictions


def predict_student(weights, image_paths, image_size):
    results = YOLO(weights).predict(source=[str(path) for path in image_paths], conf=0.001, imgsz=image_size,
                                    stream=True, verbose=False)
    return [{"boxes": result.boxes.xyxy.cpu(), "scores": result.boxes.conf.cpu(), "labels": result.boxes.cls.cpu().long()}
            for result in results]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(ROOT / "data/bdd100k"))
    parser.add_argument("--student_weights", default=str(ROOT / "ckpts/student/weights/best.pt"))
    parser.add_argument("--image_size", type=int, default=640)
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    filter_settings = read_settings(data_dir)
    filter_settings.pop("label_threshold", None)
    image_paths, sizes, targets = load_split(data_dir, "test")
    teacher = [filter_predictions(prediction, *size, **filter_settings)
               for prediction, size in zip(load_teacher_predictions(data_dir, "test", image_paths), sizes)]
    models = {"teacher": teacher, "student": predict_student(args.student_weights, image_paths, args.image_size)}
    for model_name, predictions in models.items():
        map_all, map_50, per_class = mean_average_precision(predictions, targets, CLASSES)
        print(f"{model_name} on test: mAP50-95 {map_all:.3f}  mAP50 {map_50:.3f}")
        print(f"  per-class mAP50-95: {per_class}")


if __name__ == "__main__":
    main()