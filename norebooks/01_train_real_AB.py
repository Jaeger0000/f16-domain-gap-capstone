# =====================================================================
# Kaggle Notebook — Run A (imgsz=640) & Run B (imgsz=1280)
# Real-F16 (Roboflow) training for domain-gap capstone study
#
# HOW TO USE ON KAGGLE:
#  1. Create a new Kaggle Notebook.
#  2. Settings -> Accelerator -> GPU (T4 x1 or P100).
#  3. Settings -> Internet -> ON (needed for pip + roboflow download).
#  4. Paste each "CELL" block below into a separate notebook cell, OR
#     paste the whole file into one cell (it runs top to bottom fine).
#  5. Put your Roboflow API key where indicated (or upload the dataset
#     as a Kaggle Dataset and skip the roboflow download cell).
# =====================================================================


# ------------------------ CELL 1: install ----------------------------
# Ultralytics pulls in torch/torchvision compatible with Kaggle's CUDA.
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "ultralytics==8.3.0", "roboflow"], check=True)

import torch, os, shutil, glob
from PIL import Image
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")


# ------------------------ CELL 2: get dataset -------------------------
# OPTION A — download directly from Roboflow (needs Internet ON):
#   Get your key + version from the Roboflow "Export -> show download code".
#   Replace the three placeholders below.
#
# OPTION B — if you uploaded the dataset as a Kaggle Dataset instead,
#   comment this whole block out and set DATA_DIR to the input path, e.g.
#   DATA_DIR = "/kaggle/input/your-dataset-name"

USE_ROBOFLOW_DOWNLOAD = True

if USE_ROBOFLOW_DOWNLOAD:
    from roboflow import Roboflow
    rf = Roboflow(api_key="PASTE_YOUR_ROBOFLOW_API_KEY")
    project = rf.workspace("capstone-j5t6y").project("f16-kob5w")
    # IMPORTANT: pick the version that has NO resize preprocessing.
    version = project.version(1)          # <-- change version number if needed
    dataset = version.download("yolov11")  # downloads to ./f16-kob5w-1 or similar
    DATA_DIR = dataset.location
else:
    DATA_DIR = "/kaggle/input/REPLACE-WITH-YOUR-KAGGLE-DATASET"

print("DATA_DIR =", DATA_DIR)
DATA_YAML = os.path.join(DATA_DIR, "data.yaml")
assert os.path.exists(DATA_YAML), f"data.yaml not found at {DATA_YAML}"

# Show the class names so you can verify they match your model dataset later.
with open(DATA_YAML) as f:
    print("----- data.yaml -----")
    print(f.read())


# ------------------- CELL 3: sanity-check resolution ------------------
# Confirms images are big enough for the imgsz=1280 run to be meaningful.
sizes_w, sizes_h = [], []
for f in glob.glob(os.path.join(DATA_DIR, "**", "*.jpg"), recursive=True)[:500]:
    try:
        w, h = Image.open(f).size
        sizes_w.append(w); sizes_h.append(h)
    except Exception:
        pass
if sizes_w:
    import statistics as st
    print(f"images sampled: {len(sizes_w)}")
    print(f"width  min/median/max: {min(sizes_w)} / {st.median(sizes_w)} / {max(sizes_w)}")
    print(f"height min/median/max: {min(sizes_h)} / {st.median(sizes_h)} / {max(sizes_h)}")
    if st.median(sizes_w) < 1280:
        print("WARNING: median width < 1280 -> the 1280 run will upscale. "
              "1280 results may not reflect real high-res detail.")
else:
    print("No .jpg found; check DATA_DIR / file extension (.png?).")


# ------------------------ CELL 4: training config ---------------------
from ultralytics import YOLO

# Shared, fixed across all runs for a fair controlled comparison.
COMMON = dict(
    data=DATA_YAML,
    epochs=100,
    patience=20,          # early stopping
    seed=0,               # reproducibility
    optimizer="AdamW",
    lr0=1e-3,
    weight_decay=5e-4,
    workers=4,
    project="/kaggle/working/runs",
    exist_ok=True,
    plots=True,           # saves PR curve, confusion matrix, etc.
    val=True,
)

# batch is scaled down at higher resolution to fit GPU memory.
RUNS = [
    dict(name="A_real_640",  imgsz=640,  batch=16),
    dict(name="B_real_1280", imgsz=1280, batch=4),
]


# ------------------------ CELL 5: run training ------------------------
results_summary = {}
for r in RUNS:
    print("\n" + "=" * 60)
    print("STARTING RUN:", r["name"], "imgsz =", r["imgsz"], "batch =", r["batch"])
    print("=" * 60)
    model = YOLO("yolo11s.pt")      # COCO-pretrained, fine-tuned on F16
    res = model.train(name=r["name"], imgsz=r["imgsz"], batch=r["batch"], **COMMON)

    # Validate the best checkpoint on the dataset's own val split.
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

print("\n\n==== SUMMARY (val split of Roboflow real-F16) ====")
for k, v in results_summary.items():
    print(k, v)


# ------------------------ CELL 6: keep outputs ------------------------
# Kaggle saves anything under /kaggle/working as a downloadable output.
# Zip the runs folder so you can download weights + plots + results.csv.
shutil.make_archive("/kaggle/working/runs_export", "zip", "/kaggle/working/runs")
print("Done. Download /kaggle/working/runs_export.zip from the Output tab.")
print("Inside you will find, per run: weights/best.pt, results.csv, "
      "PR_curve.png, confusion_matrix.png, etc.")
