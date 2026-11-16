# Open-vocabulary Auto-labeling for Autonomous Driving

### TODO
<!-- 
### This repo uses a VLM / open-vocabulary detector as a teacher to generate pseudo-labels, then train a lightweight student detector.
-->
*(Work in Progress: This repository is currently under active development.)*

## Introduction

TODO: teacher model generates pseudo-labels images from text prompts, then a student model trains on those labels. 

```bash
python pseudo_label.py --ratio 0.03

```

```bash
python train_student.py --epochs 30
python evaluate.py
```