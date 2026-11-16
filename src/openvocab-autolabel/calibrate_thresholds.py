import argparse
from pathlib import Path

from utils.bdd100k import CLASSES
from utils.data_files import load_split, load_teacher_predictions, write_class_thresholds
from utils.paths import ROOT
from utils.pseudo_labels import calibrate_thresholds, filter_predictions

CANDIDATE_THRESHOLDS = [round(0.1 + 0.025 * step, 3) for step in range(21)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(ROOT / "data/bdd100k"))
    parser.add_argument("--num_calibration", type=int, default=200)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--min_targets", type=int, default=10)
    parser.add_argument("--default_threshold", type=float, default=0.35)
    parser.add_argument("--output", default=str(ROOT / "output/calibration/class_thresholds.yaml"))
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    image_paths, sizes, targets = load_split(data_dir, "train")
    image_paths, sizes, targets = image_paths[:args.num_calibration], sizes[:args.num_calibration], targets[:args.num_calibration]
    filtered = [filter_predictions(prediction, *size)
                for prediction, size in zip(load_teacher_predictions(data_dir, "train", image_paths), sizes)]
    thresholds, report = calibrate_thresholds(filtered, targets, CLASSES, CANDIDATE_THRESHOLDS, args.beta,
                                              args.min_targets, args.default_threshold)

    print(f"calibrated on {len(image_paths)} human-labeled train images (F{args.beta:g})")
    print(f"  {'class':<15}{'targets':>8}{'threshold':>11}{'precision':>11}{'recall':>8}")
    for class_name, row in report.items():
        print(f"  {class_name:<15}{row['targets']:>8}{row['threshold']:>11.3f}"
              f"{row.get('precision', float('nan')):>11.3f}{row.get('recall', float('nan')):>8.3f}  {row.get('note', '')}")
    write_class_thresholds(Path(args.output), thresholds, len(image_paths), report)
    print("thresholds saved to:", args.output)


if __name__ == "__main__":
    main()