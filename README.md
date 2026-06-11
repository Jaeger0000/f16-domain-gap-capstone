# f16-domain-gap-capstone

Code for a capstone experiment on whether an F16 detector trained on real aircraft photos transfers to physical scale-model aircraft. Short answer: it does not, unless you train on the models too.

> Full methodology, tables, and analysis live in the capstone report (PDF/DOCX submitted on Canvas). This README only covers the code.

## What's here

Four base detectors plus two control runs, all YOLO11m, same hyperparameters:

| Run | Trained on | imgsz |
|-----|-----------|-------|
| A | real photos | 640 |
| B | real photos | 1280 |
| C | scale models | 640 |
| D | scale models | 1280 |
| E | real, capped to 547 imgs | 640 |
| F | real + model mixed | 640 |

Everything is cross-checked on one untouched scale-model test set, so the numbers are comparable. Headline: real-trained runs score near zero on models; model-trained runs don't.

## Layout

```
notebooks/   training + evaluation (Kaggle .py exports)
configs/     dataset yaml files
scripts/     frame-extraction notes
results/     metrics json + figures
```

| File | Does |
|------|------|
| `notebooks/01_train_real_AB.py` | runs A, B |
| `notebooks/02_train_model_CD.py` | runs C, D |
| `notebooks/03_helper_autolabel.py` | exports the helper model used to pre-suggest boxes |
| `notebooks/04_cross_evaluation.py` | scores every checkpoint on the model test set |
| `notebooks/05_control_experiments.py` | runs E and F |
| `notebooks/06_failure_survey.py` | sweeps the whole test set, flags misses |

## Running it

Built for Kaggle (2x T4). Per notebook:

1. attach the two datasets (real + scale-model) and, for eval, the trained weights
2. set a GPU accelerator
3. edit the paths in the first config cell
4. run top to bottom

Weights from one notebook get uploaded as a Kaggle Dataset and pointed at by the next. The 1280 model run uses a single GPU (the 2-GPU path stalls at that resolution).

## Data

- **Real:** a public Roboflow F16 set (CC BY 4.0) - https://universe.roboflow.com/arsivs-workspace/f16-kob5w-cazuz
- **Scale-model:** recorded by the author, frames pulled with [video_to_png_convertor](https://github.com/Jaeger0000/video_to_png_convertor), boxed by hand in X-AnyLabeling (a helper model suggested boxes, all of them checked manually). Split is done per-video so no clip leaks across train/val/test.

## Notes

- single class (`F16`)
- background-only frames are kept in training to keep false positives down
- seeds fixed for repeatability

## License

MIT for the code. Real dataset stays under its original CC BY 4.0.
