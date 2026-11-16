import argparse
import io
import shutil
from collections import Counter
from pathlib import Path

from huggingface_hub import HfFileSystem
from PIL import Image
from tqdm import tqdm

from utils.bdd100k import CLASS_SYNONYMS, CLASSES, SPLITS, read_human_labels, read_image_bytes, stream_images
from utils.formats import to_scored_line, to_yolo_line
from utils.paths import ROOT
from utils.teacher import build_class_prompt, detect_classes, load_teacher

OUTPUT_KINDS = ["images", "labels_human", "predictions_teacher"]


def write_lines(path, lines):
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", default="dronefreak/BDD100K")
    parser.add_argument("--model_id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--ratio", type=float, default=0.03)
    parser.add_argument("--no_synonyms", action="store_true")
    parser.add_argument("--teacher_floor", type=float, default=0.05)
    parser.add_argument("--output", default=str(ROOT / "data/bdd100k"))
    args = parser.parse_args()

    model, processor = load_teacher(args.model_id)
    class_prompt = build_class_prompt(processor, CLASSES, None if args.no_synonyms else CLASS_SYNONYMS)
    print("prompt:", class_prompt["prompt"])
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

            boxes, scores, class_ids = detect_classes(model, processor, image, class_prompt, args.teacher_floor)
            write_lines(folders["predictions_teacher"] / f"{name}.txt",
                        [to_scored_line(class_id, score, box) for class_id, score, box in zip(class_ids, scores, boxes)])

        print(f"{split} human boxes per class:", dict(human_counts.most_common()))


if __name__ == "__main__":
    main()