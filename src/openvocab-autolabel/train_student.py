import argparse
from pathlib import Path

from ultralytics import YOLO

from utils.paths import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data/bdd100k/data.yaml"))
    parser.add_argument("--model", default=str(ROOT / "ckpts/yolo11n.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--image_size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--project", default=str(ROOT / "ckpts"))
    parser.add_argument("--name", default="student")
    args = parser.parse_args()

    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.image_size, batch=args.batch,
                project=str(Path(args.project).resolve()), name=args.name, exist_ok=True)
    print("weights saved to:", Path(model.trainer.save_dir) / "weights" / "best.pt")


if __name__ == "__main__":
    main()