# Open-vocabulary Auto-labeling

### This repo uses a VLM / open-vocabulary detector as a teacher to generate pseudo-labels, then train a lightweight student detector.

*(Work in Progress: This repository is currently under active development.)*

## Introduction

TODO: teacher model generates pseudo-labels images from text prompts, then a student model trains on those labels. 

```bash
python run_teacher.py --ratio 0.03

python calibrate_thresholds.py --num_calibration 200
python analyze_pseudo_labels.py
python make_pseudo_labels.py --class_thresholds ../../output/calibration/class_thresholds.yaml
```

```bash
python train_student.py --epochs 30
python evaluate.py
```