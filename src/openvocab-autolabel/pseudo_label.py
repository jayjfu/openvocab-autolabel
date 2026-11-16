import argparse
import io
import shutil
from collections import Counter
from pathlib import Path

from huggingface_hub import HfFileSystem
from PIL import Image
from tqdm import tqdm

from utils.bdd100k import CLASSES, read_human_labels, read_image_bytes, stream_images
from utils.paths import ROOT
from utils.formats import to_scored_line, to_yolo_line, write_data_config
from utils.teacher import build_prompt, detect, label_to_class, load_teacher

SPLITS = ["train", "val", "test"]
OUTPUT_KINDS = ["images", "labels", "labels_human", "predictions_teacher"]


def write_lines(path, lines):
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="dronefreak/BDD100K")
    parser.add_argument("--model_id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--ratio", type=float, default=0.03)
    parser.add_argument("--teacher_floor", type=float, default=0.1)
    parser.add_argument("--label_threshold", type=float, default=0.35)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--output", default=str(ROOT / "data/bdd100k"))
    args = parser.parse_args()

    model, processor = load_teacher(args.model_id)
    prompt = build_prompt(CLASSES)
    file_system = HfFileSystem()
    output_dir = Path(args.output).resolve()

    for split in SPLITS:
        folders = {kind: output_dir / kind / split for kind in OUTPUT_KINDS}
        for folder in folders.values():
            shutil.rmtree(folder, ignore_errors=True)
            folder.mkdir(parents=True)

        human_counts = Counter()
        rows, count = stream_images(args.repo_id, split, args.ratio)
        for row in tqdm(rows, total=count, desc=split):
            image_path = row["image"]["path"]
            name = Path(image_path).stem
            image_bytes = read_image_bytes(file_system, row["image"])
            (folders["images"] / f"{name}.jpg").write_bytes(image_bytes)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            width, height = image.size

            human_boxes, human_classes = read_human_labels(file_system, image_path)
            human_counts.update(CLASSES[class_id] for class_id in human_classes)
            write_lines(folders["labels_human"] / f"{name}.txt",
                        [to_yolo_line(class_id, box, width, height) for box, class_id in zip(human_boxes, human_classes)])

            boxes, scores, labels = detect(model, processor, image, prompt, args.teacher_floor, args.text_threshold)
            predictions = [(label_to_class(label, CLASSES), score, box) for box, score, label in zip(boxes, scores, labels)]
            predictions = [prediction for prediction in predictions if prediction[0] is not None]
            write_lines(folders["predictions_teacher"] / f"{name}.txt",
                        [to_scored_line(class_id, score, box) for class_id, score, box in predictions])
            write_lines(folders["labels"] / f"{name}.txt",
                        [to_yolo_line(class_id, box, width, height)
                         for class_id, score, box in predictions if score >= args.label_threshold])

        print(f"{split} human boxes per class:", dict(human_counts.most_common()))

    write_data_config(output_dir, CLASSES)


if __name__ == "__main__":
    main()