import json

from datasets import Image as ImageFeature
from datasets import load_dataset
from huggingface_hub import HfFileSystem

CLASSES = ["person", "rider", "car", "truck", "bus", "train", "motor", "bike", "traffic light", "traffic sign"]
CLASS_SYNONYMS = {
    "person": ["pedestrian"],
    "rider": ["cyclist"],
    "train": ["tram"],
    "motor": ["motorcycle", "motorbike"],
    "bike": ["bicycle"],
    "traffic sign": ["road sign"],
}
SPLITS = ["train", "val", "test"]
IMAGE_ROOT = "data/images_100k/100k"
LABEL_ROOT = "data/labels/100k"


def stream_images(repo_id, split, ratio):
    split_dir = f"datasets/{repo_id}/{IMAGE_ROOT}/{split}"
    num_images = len(HfFileSystem().glob(f"{split_dir}/shard_*/*.jpg"))
    count = max(1, round(num_images * ratio))
    dataset = load_dataset("imagefolder", data_files={split: f"hf://{split_dir}/shard_*/*"}, split=split, streaming=True)
    return dataset.take(count).cast_column("image", ImageFeature(decode=False)), count


def read_image_bytes(file_system, image_record):
    return image_record["bytes"] or file_system.read_bytes(image_record["path"])


def read_human_labels(file_system, image_path):
    label_path = image_path.replace(f"/{IMAGE_ROOT}/", f"/{LABEL_ROOT}/").rsplit(".", 1)[0] + ".json"
    record = json.loads(file_system.read_text(label_path))
    boxes, class_ids = [], []
    for frame in record.get("frames", []):
        for labeled_object in frame.get("objects", []):
            category = labeled_object.get("category")
            box = labeled_object.get("box2d")
            if box is None or category not in CLASSES:
                continue
            boxes.append([box["x1"], box["y1"], box["x2"], box["y2"]])
            class_ids.append(CLASSES.index(category))
    return boxes, class_ids