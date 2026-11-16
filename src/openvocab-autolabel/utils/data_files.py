import torch
import yaml
from PIL import Image

SETTINGS_FILE = "pseudo_label_settings.yaml"


def to_yolo_line(class_id, box, width, height):
    x1, y1, x2, y2 = box
    return (f"{class_id} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
            f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")


def to_scored_line(class_id, score, box):
    return f"{class_id} {score:.4f} " + " ".join(f"{coord:.1f}" for coord in box)


def write_lines(path, lines):
    path.write_text("\n".join(lines))


def read_rows(path):
    return [line.split() for line in path.read_text().splitlines() if line.strip()]


def read_yolo_boxes(path, width, height):
    rows = read_rows(path)
    centers = torch.tensor([[float(value) for value in row[1:]] for row in rows]).reshape(-1, 4)
    centers *= torch.tensor([width, height, width, height])
    boxes = torch.cat([centers[:, :2] - centers[:, 2:] / 2, centers[:, :2] + centers[:, 2:] / 2], dim=1)
    return {"boxes": boxes, "labels": torch.tensor([int(row[0]) for row in rows], dtype=torch.long)}


def read_scored_boxes(path):
    rows = read_rows(path)
    return {"labels": torch.tensor([int(row[0]) for row in rows], dtype=torch.long),
            "scores": torch.tensor([float(row[1]) for row in rows]),
            "boxes": torch.tensor([[float(value) for value in row[2:]] for row in rows]).reshape(-1, 4)}


def load_split(data_dir, split):
    image_paths = sorted((data_dir / "images" / split).glob("*.jpg"))
    sizes = [Image.open(path).size for path in image_paths]
    targets = [read_yolo_boxes(data_dir / "labels_human" / split / f"{path.stem}.txt", *size)
               for path, size in zip(image_paths, sizes)]
    return image_paths, sizes, targets


def load_teacher_predictions(data_dir, split, image_paths):
    return [read_scored_boxes(data_dir / "predictions_teacher" / split / f"{path.stem}.txt") for path in image_paths]


def write_data_config(data_dir, class_names):
    data_config = {"path": str(data_dir), "train": "images/train", "val": "images/val", "test": "images/test",
                   "names": dict(enumerate(class_names))}
    (data_dir / "data.yaml").write_text(yaml.safe_dump(data_config, sort_keys=False))


def write_settings(data_dir, settings):
    (data_dir / SETTINGS_FILE).write_text(yaml.safe_dump(settings, sort_keys=False))


def read_settings(data_dir):
    path = data_dir / SETTINGS_FILE
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def write_class_thresholds(path, thresholds, num_calibration_images, report):
    content = {"num_calibration_images": num_calibration_images, "thresholds": thresholds, "report": report}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(content, sort_keys=False))


def read_class_thresholds(path):
    return yaml.safe_load(path.read_text())