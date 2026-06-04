# =====================================================================
# Kaggle Notebook — Auto-Label HELPER model
# Trains a quick YOLO11s on the ~187 hand-labeled model-F16 frames.
# Purpose: use this model inside X-AnyLabeling to pre-label the
# remaining frames. This is NOT one of the report's A/B/C/D models.
#
# SETUP ON KAGGLE:
#  - Accelerator: GPU (T4 x1 is enough for this small/quick run)
#  - Internet: ON
#  - Upload your dataset as a Kaggle Dataset named "f16-dataset"
#    so it lands at /kaggle/input/f16-dataset/dataset
# =====================================================================


# ------------------------ CELL 1: setup ------------------------------
import subprocess, sys, os, shutil
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "-U", "ultralytics"], check=True)

import numpy as np, torch
print("numpy:", np.__version__, "| torch:", torch.__version__, "| CUDA:", torch.cuda.is_available())
from ultralytics import settings
settings.update({"wandb": False})
print("ultralytics ready")


# ------------------- CELL 2: fix the data.yaml -----------------------
# Your uploaded data.yaml uses names:['f16'] and points at a fixed path.
# We rewrite a clean copy in /kaggle/working so:
#   (1) class name is 'F16'  -> matches the Roboflow real dataset exactly
#   (2) path is correct for the Kaggle input mount
# Adjust DATASET_ROOT if your upload landed somewhere else (check the
# right-hand "Input" panel for the exact folder name).

DATASET_ROOT = "/kaggle/input/f16-dataset/dataset"   # <-- verify in Input panel
assert os.path.exists(DATASET_ROOT), f"Not found: {DATASET_ROOT} (check Input panel name)"

data_yaml = f"""\
path: {DATASET_ROOT}
train: images/train
val: images/val
nc: 1
names: ['F16']
"""
DATA_YAML = "/kaggle/working/helper_data.yaml"
with open(DATA_YAML, "w") as f:
    f.write(data_yaml)
print("Wrote:", DATA_YAML)
print(data_yaml)

# quick sanity: count images/labels
import glob
for split in ["train", "val"]:
    imgs = glob.glob(os.path.join(DATASET_ROOT, "images", split, "*"))
    lbls = glob.glob(os.path.join(DATASET_ROOT, "labels", split, "*.txt"))
    print(f"{split}: {len(imgs)} images, {len(lbls)} labels")


# ------------------- CELL 3: train helper model ----------------------
from ultralytics import YOLO

model = YOLO("yolo11s.pt")   # small + fast; good enough as a labeling aid

results = model.train(
    data=DATA_YAML,
    epochs=80,
    patience=20,          # stops early once val plateaus
    imgsz=960,            # your frames are 1920x1080; 960 keeps detail, trains fast
    batch=16,
    seed=0,
    optimizer="AdamW",
    lr0=1e-3,
    device=0,             # single T4 is fine for a helper model
    project="/kaggle/working/runs",
    name="helper_autolabel",
    exist_ok=True,
    plots=True,
)

# validate best checkpoint
metrics = model.val(data=DATA_YAML, imgsz=960,
                    project="/kaggle/working/runs",
                    name="helper_autolabel_val", exist_ok=True)
print("Helper model metrics:",
      {"mAP50": float(metrics.box.map50),
       "mAP50-95": float(metrics.box.map),
       "P": float(metrics.box.mp),
       "R": float(metrics.box.mr)})


# ------------------- CELL 4: export ONNX for X-AnyLabeling -----------
# X-AnyLabeling loads YOLO models as ONNX. We export best.pt -> best.onnx
# with a fixed input size and opset 12 (widely compatible with XAL).
best = "/kaggle/working/runs/helper_autolabel/weights/best.pt"
print("Best weights at:", best, "exists:", os.path.exists(best))

export_model = YOLO(best)
onnx_path = export_model.export(
    format="onnx",
    imgsz=960,        # MUST match what you'll set in X-AnyLabeling
    opset=12,         # XAL-friendly opset
    simplify=True,    # cleaner graph
    dynamic=False,    # fixed input size -> more reliable in XAL
)
print("ONNX exported to:", onnx_path)   # -> .../weights/best.onnx

# ---- Build the YAML config X-AnyLabeling needs alongside the .onnx ----
# X-AnyLabeling custom YOLO models expect a small .yaml describing the model.
xal_cfg = """\
type: yolo11
name: helper_f16_autolabel
display_name: F16 Helper (auto-label)
model_path: best.onnx
input_width: 960
input_height: 960
score_threshold: 0.25
nms_threshold: 0.45
confidence_threshold: 0.25
classes:
  - F16
"""
cfg_path = "/kaggle/working/runs/helper_autolabel/weights/helper_f16.yaml"
with open(cfg_path, "w") as f:
    f.write(xal_cfg)
print("XAL config written to:", cfg_path)

# ---- Zip just the two files you need for X-AnyLabeling ----
import os as _os
xal_dir = "/kaggle/working/xal_model"
_os.makedirs(xal_dir, exist_ok=True)
shutil.copy(_os.path.join(_os.path.dirname(best), "best.onnx"),
            _os.path.join(xal_dir, "best.onnx"))
shutil.copy(cfg_path, _os.path.join(xal_dir, "helper_f16.yaml"))
shutil.make_archive("/kaggle/working/xal_model", "zip", xal_dir)

# Also keep the full run (plots, results.csv) for the report later.
shutil.make_archive("/kaggle/working/helper_export", "zip",
                    "/kaggle/working/runs/helper_autolabel")

print("\nDownload from Output tab:")
print("  xal_model.zip      <-- best.onnx + helper_f16.yaml for X-AnyLabeling")
print("  helper_export.zip  <-- full run (plots, results.csv) for records")
