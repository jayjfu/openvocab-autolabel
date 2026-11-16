import torch
import yaml


def to_yolo_line(class_id, box, width, height):
    x1, y1, x2, y2 = box
    return (f"{class_id} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
            f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}")


def to_scored_line(class_id, score, box):
    return f"{class_id} {score:.4f} " + " ".join(f"{coord:.1f}" for coord in box)


def read_rows(path):
    return [line.split() for line in path.read_text().splitlines() if line.strip()]


def read_yolo_boxes(path, width, height):
    rows = read_rows(path)
    centers = torch.tensor([[float(value) for value in row[1:]] for row in rows]).reshape(-1, 4)
    centers *= torch.tensor([width, height, width, height])
    boxes = torch.cat([centers[:, :2] - centers[:, 2:] / 2, centers[:, :2] + centers[:, 2:] / 2], dim=1)
    return {"boxes": boxes, "labels": torch.tensor([int(row[0]) for row in rows], dtype=torch.long)}


def read_scored_boxes(path, min_score=0.0):
    rows = [row for row in read_rows(path) if float(row[1]) >= min_score]
    return {"labels": torch.tensor([int(row[0]) for row in rows], dtype=torch.long),
            "scores": torch.tensor([float(row[1]) for row in rows]),
            "boxes": torch.tensor([[float(value) for value in row[2:]] for row in rows]).reshape(-1, 4)}


def write_data_config(output_dir, class_names):
    data_config = {"path": str(output_dir), "train": "images/train", "val": "images/val", "test": "images/test",
                   "names": dict(enumerate(class_names))}
    (output_dir / "data.yaml").write_text(yaml.safe_dump(data_config, sort_keys=False))
