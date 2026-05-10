# Datasets

This directory is intentionally kept empty in the repository. Follow the steps
below to populate it before running `src/fairness_evaluation.py`.

## 1. UCI Heart Failure Prediction

Download `heart.csv` from Kaggle and place it at `data/heart.csv`:

- Source: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
- Expected location: `data/heart.csv`
- Size: ~30 KB

## 2. PAPILA Retinal Fundus Dataset

Run the provided helper to fetch, extract, and pre-process the archive:

```bash
python src/download_papila.py
```

This will create:

- `data/papila/clinical_data.csv` — per-patient metadata (ID, Gender, Age, Diagnosis)
- `data/papila/images/` — fundus images (RET<ID>OD.jpg, RET<ID>OS.jpg)

Source: Kovalyk et al., 2022 — https://doi.org/10.6084/m9.figshare.14798004.v2
