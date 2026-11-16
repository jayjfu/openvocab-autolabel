import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from utils.bdd100k import CLASSES
from utils.paths import ROOT
from utils.teacher import build_prompt, detect, load_teacher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model_id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--prompt", default=build_prompt(CLASSES))
    parser.add_argument("--box_threshold", type=float, default=0.35)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--output", default=str(ROOT / "output/detections.jpg"))
    args = parser.parse_args()

    model, processor = load_teacher(args.model_id)
    image = Image.open(args.image).convert("RGB")
    boxes, scores, labels = detect(model, processor, image, args.prompt, args.box_threshold, args.text_threshold)

    draw = ImageDraw.Draw(image)
    detections = sorted(zip(boxes, scores, labels), key=lambda detection: -detection[1])
    for box, score, label in detections:
        print(f"{score:.3f}  {label!r:<24} {[round(coord) for coord in box]}")
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0] + 3, box[1] + 2), f"{label} {score:.2f}", fill="yellow")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)


if __name__ == "__main__":
    main()