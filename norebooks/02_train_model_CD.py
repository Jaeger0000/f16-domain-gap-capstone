# =====================================================================
# Kaggle Notebook — Run C (imgsz=640) & Run D (imgsz=1280)
# Model-F16 (maket) dataset training for the domain-gap capstone study.
# Uses IDENTICAL settings to the Roboflow A/B runs for a fair comparison.
#
# SETUP ON KAGGLE:
#  - Accelerator: GPU T4 x2 (same as A/B)
#  - Internet: ON
#  - Upload final_dataset as a Kaggle Dataset.
#    Verify its mounted path in the right-hand "Input" panel and set
#    DATASET_ROOT below accordingly.
# =====================================================================


# ------------------------ CELL 1: setup ------------------------------
import subprocess, sys, os, shutil, glob
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "-U", "ultralytics"], check=True)

import numpy as np, torch
print("numpy:", np.__version__, "| torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
from ultralytics import settings
settings.update({"wandb": False})
print("ultralytics ready")


# ------------------- CELL 2: fix + write data.yaml -------------------
# Your dataset uses names:['f16']; we rewrite to ['F16'] so the class
# name/index matches the Roboflow real dataset EXACTLY (required for the
# cross-evaluation step where real-trained models are tested here).
#
# IMPORTANT: verify DATASET_ROOT against the Input panel. The folder that
# directly CONTAINS train/ val/ test/ (or images/labels) is the root.

DATASET_ROOT = "/kaggle/input/final-dataset"   # <-- VERIFY in Input panel

assert os.path.exists(DATASET_ROOT), f"Not found: {DATASET_ROOT} (check Input panel)"
print("Contents of DATASET_ROOT:")
for x in sorted(os.listdir(DATASET_ROOT)):
    print("  ", x)

# Detect the layout. Two common cases:
#   (a) DATASET_ROOT/{train,val,test}/{images,labels}
#   (b) DATASET_ROOT/images/{train,val,test} + labels/{train,val,test}
def detect_paths(root):
    if os.path.isdir(os.path.join(root, "train", "images")):
        return ("train/images", "val/images", "test/images")
    if os.path.isdir(os.path.join(root, "images", "train")):
        return ("images/train", "images/val", "images/test")
    # fallback: assume case (a)
    return ("train/images", "val/images", "test/images")

tr, va, te = detect_paths(DATASET_ROOT)

data_yaml = f"""\
path: {DATASET_ROOT}
train: {tr}
val: {va}
test: {te}
nc: 1
names: ['F16']
"""
DATA_YAML = "/kaggle/working/model_data.yaml"
with open(DATA_YAML, "w") as f:
    f.write(data_yaml)
print("\nWrote data.yaml:\n", data_yaml)

# sanity counts (images + how many labels are non-empty = positives)
for split_path, label in [(tr, "train"), (va, "val"), (te, "test")]:
    img_dir = os.path.join(DATASET_ROOT, split_path)
    lbl_dir = img_dir.replace("images", "labels")
    imgs = glob.glob(os.path.join(img_dir, "*"))
    lbls = glob.glob(os.path.join(lbl_dir, "*.txt"))
    nonempty = sum(1 for p in lbls if os.path.getsize(p) > 0)
    print(f"{label}: {len(imgs)} imgs | {len(lbls)} labels | {nonempty} positive, {len(lbls)-nonempty} negative")


# ------------------------ CELL 3: training config --------------------
# EXACTLY matches the Roboflow A/B COMMON config for a fair comparison.
from ultralytics import YOLO

COMMON = dict(
    data=DATA_YAML,
    epochs=100,
    patience=20,
    seed=0,
    optimizer="AdamW",
    lr0=1e-3,
    weight_decay=5e-4,
    workers=4,
    device=[0, 1],        # 2x T4, same as A/B
    project="/kaggle/working/runs",
    exist_ok=True,
    plots=True,
    val=True,
)

RUNS = [
    dict(name="C_model_640",  imgsz=640,  batch=32),
    dict(name="D_model_1280", imgsz=1280, batch=8),
]


# ------------------------ CELL 4: run training -----------------------
results_summary = {}
for r in RUNS:
    print("\n" + "=" * 60)
    print("STARTING RUN:", r["name"], "imgsz =", r["imgsz"], "batch =", r["batch"])
    print("=" * 60)
    model = YOLO("yolo11m.pt")        # same model family as A/B
    model.train(name=r["name"], imgsz=r["imgsz"], batch=r["batch"], **COMMON)

    # validate best checkpoint on this dataset's OWN val split
    metrics = model.val(name=r["name"] + "_val", imgsz=r["imgsz"],
                        data=DATA_YAML, project="/kaggle/working/runs",
                        exist_ok=True)
    results_summary[r["name"]] = {
        "mAP50":    float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall":    float(metrics.box.mr),
    }
    print(r["name"], "->", results_summary[r["name"]])

print("\n\n==== SUMMARY (val split of model-F16 dataset) ====")
for k, v in results_summary.items():
    print(k, v)


# ------------------------ CELL 5: export -----------------------------
shutil.make_archive("/kaggle/working/runs_CD_export", "zip", "/kaggle/working/runs")
print("Done. Download /kaggle/working/runs_CD_export.zip from the Output tab.")
print("Per run: weights/best.pt, results.csv, PR_curve.png, confusion_matrix.png")
print("KEEP the best.pt files — needed for cross-evaluation (domain gap).")
