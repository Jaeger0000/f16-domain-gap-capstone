# Real-to-Model Domain Gap in F16 Detection

A Cross-Domain YOLO11 Comparison for Air-Defense Scale-Model Targets

This repository contains the implementation for a capstone project investigating the **real-to-model domain gap** in single-class F16 detection: whether a detector trained on *real* aircraft imagery can generalize to *physical scale-model* targets (as used in TEKNOFEST-style air-defense scenarios), and how much target-domain training closes that gap.

**Author:** Oğuzhan Ceviz
**Supervisor:** Asst. Prof. Rıfat Kurban
**Institution:** Abdullah Gül University — Computer Engineering

---

## TL;DR — Key Result

Four identical YOLO11m detectors were trained (two data domains × two input resolutions) and evaluated on a common, held-out **scale-model** test set:

| Model | Train data | imgsz | mAP50 (scale-model test) |
|-------|-----------|-------|--------------------------|
| A | Real F16 (Roboflow) | 640 | **0.0003** |
| B | Real F16 (Roboflow) | 1280 | **0.0002** |
| C | Scale-model (ours) | 640 | 0.871 |
| D | Scale-model (ours) | 1280 | **0.964** |

A detector trained only on real aircraft imagery (≈0.96 mAP50 on real data) **collapses to ~0** on physical scale-model targets — a **~96-point real-to-model domain gap**. Training on target-domain data restores near-ceiling accuracy.

---

## Repository Structure

```
.
├── README.md
├── notebooks/
│   ├── 01_train_real_AB.ipynb        # Runs A & B: YOLO11m on real-F16 (Roboflow), imgsz 640 / 1280
│   ├── 02_train_model_CD.ipynb       # Runs C & D: YOLO11m on scale-model dataset, imgsz 640 / 1280
│   ├── 03_helper_autolabel.ipynb     # Helper model for X-AnyLabeling assisted annotation
│   └── 04_cross_evaluation.ipynb     # Evaluate all models on the common scale-model test set
├── configs/
│   ├── real_data.yaml                # Roboflow real-F16 data config (names: ['F16'])
│   └── model_data.yaml               # Scale-model dataset config (names: ['F16'])
├── scripts/
│   └── extract_frames.md             # How frames were extracted (see Data Preparation below)
└── results/
    ├── crosseval_results.json        # Cross-domain metrics (Table 5 of the report)
    └── figures/                      # Confusion matrices, PR curves, qualitative detections
```

> The `.py` files exported from the Kaggle notebooks are included; you can paste each into a Kaggle notebook cell or run them directly in an equivalent environment.

---

## Experimental Setup

All four runs share an identical configuration for a fair comparison:

- **Model:** YOLO11m (Ultralytics), COCO-pretrained, fine-tuned for single-class `F16`
- **Optimizer:** AdamW, `lr0=1e-3`, `weight_decay=5e-4`
- **Epochs:** 100, early stopping `patience=20`, `seed=0`
- **Resolution:** 640 (A, C) / 1280 (B, D)
- **Hardware:** 2× NVIDIA Tesla T4 (Kaggle)
- **Framework:** Ultralytics YOLO on PyTorch (CUDA)

---

## Datasets

### Real F16 (source domain)
Public Roboflow Universe dataset (CC BY 4.0): 1,768 real F16 images (1,446 train / 179 val / 143 test).
Source: https://universe.roboflow.com/arsivs-workspace/f16-kob5w-cazuz

### Scale-Model F16 (target domain, self-captured)
Self-recorded videos of physical scale-model F16 targets at varied distances (5–17 m), backgrounds, and lighting. Total **972 frames**, split at the **video level** to prevent temporal leakage:

| Split | Images | Positive | Negative |
|-------|--------|----------|----------|
| Train | 715 | 547 | 168 |
| Val   | 123 | 123 | 0 |
| Test  | 134 | 132 | 2 |

The test set is drawn from **videos entirely excluded** from training and validation.

---

## Data Preparation

1. **Frame extraction** — videos were converted to frames using the
   [`video_to_png_convertor`](https://github.com/Jaeger0000/video_to_png_convertor) tool.
2. **Annotation** — all frames were annotated and verified manually in
   [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) on a local machine. A helper YOLO
   model (`03_helper_autolabel.ipynb`) was used only to pre-suggest boxes; **every single label
   was manually reviewed and corrected by the author**, so the final annotations (including the
   test set used for evaluation) are human-verified ground truth. Single class `F16`; negative
   (background) frames included.
3. **Video-level split** — frames from any single video appear in only one split.

---

## How to Reproduce

1. Open the notebooks in `notebooks/` on Kaggle (GPU enabled, Internet on).
2. Run `01_train_real_AB` and `02_train_model_CD` to produce the four `best.pt` checkpoints.
3. Run `04_cross_evaluation` with all four weights to reproduce the domain-gap table and
   qualitative figures.

---

## Acknowledgements

AI tools were used to help draft documentation, debug training/evaluation scripts, and format
results. All experimental design, data collection and annotation, model training, and result
interpretation were performed and verified by the author.

## License

Code released under the MIT License. The real-F16 dataset is © its respective authors under CC BY 4.0.
