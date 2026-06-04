# =====================================================================
# Kaggle Notebook — CROSS-EVALUATION
# Evaluates ALL trained models on the SAME held-out scale-model TEST set.
# This produces the domain-gap table (the core result of the report).
#
# SETUP:
#  - GPU T4 x1, Internet ON
#  - Have available on Kaggle:
#      * the model-F16 dataset (with test split)  -> for the test images/labels
#      * each model's best.pt: A, B, C, D (+ optional synthetic S)
#    Upload the best.pt files as a Kaggle Dataset (e.g. "f16-weights")
#    OR keep them in /kaggle/working if produced in the same session.
# =====================================================================

# ------------------------ CELL 1: setup ------------------------------
import subprocess, sys, os, glob, json
os.environ["WANDB_DISABLED"] = "true"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "ultralytics"], check=True)
import torch
from ultralytics import YOLO, settings
settings.update({"wandb": False})
print("CUDA:", torch.cuda.is_available())


# ------------------- CELL 2: point to test data + weights ------------
# The model-F16 dataset root (must contain images/test + labels/test).
DATASET_ROOT = "/kaggle/input/datasets/ouzhanceviz/final-dataset/final_dataset"  # verify

# Write a data.yaml whose 'val' points at the TEST split, so model.val()
# evaluates on the held-out scale-model test set.
TEST_YAML = "/kaggle/working/test_eval.yaml"
with open(TEST_YAML, "w") as f:
    f.write(f"path: {DATASET_ROOT}\ntrain: images/train\nval: images/test\nnc: 1\nnames: ['F16']\n")
print("Test-eval yaml written.\n")

# Map model name -> path to its best.pt.  EDIT these paths to where your
# weights actually are (check the Input panel / working dir).
WEIGHTS = {
    "A_real_640":   "/kaggle/input/f16-weights/A_real_640.pt",
    "B_real_1280":  "/kaggle/input/f16-weights/B_real_1280.pt",
    "C_model_640":  "/kaggle/input/f16-weights/C_model_640.pt",
    "D_model_1280": "/kaggle/input/f16-weights/D_model_1280.pt",
    # "S_synth_1280": "/kaggle/input/f16-weights/S_synth_1280.pt",  # optional
}
# imgsz used per model (match training resolution)
IMGSZ = {"A_real_640":640, "B_real_1280":1280, "C_model_640":640,
         "D_model_1280":1280, "S_synth_1280":1280}

# sanity: confirm each weight exists
for k, p in WEIGHTS.items():
    print(f"{k}: {'OK' if os.path.exists(p) else 'MISSING -> fix path'}  ({p})")


# ------------------- CELL 3: cross-evaluate on test ------------------
results = {}
for name, wpath in WEIGHTS.items():
    if not os.path.exists(wpath):
        print(f"SKIP {name} (weight missing)"); continue
    print("\n=== Evaluating", name, "on scale-model TEST set ===")
    model = YOLO(wpath)
    m = model.val(
        data=TEST_YAML,
        imgsz=IMGSZ[name],
        split="val",                 # 'val' here = the test images (see yaml)
        project="/kaggle/working/crosseval",
        name=name + "_on_modeltest",
        exist_ok=True,
        plots=True,                  # saves confusion matrix + PR curve on TEST
    )
    results[name] = {
        "mAP50":    round(float(m.box.map50), 4),
        "mAP50-95": round(float(m.box.map), 4),
        "precision":round(float(m.box.mp), 4),
        "recall":   round(float(m.box.mr), 4),
    }
    print(name, "->", results[name])

print("\n\n==== CROSS-DOMAIN RESULTS (all models on scale-model TEST set) ====")
print(json.dumps(results, indent=2))
with open("/kaggle/working/crosseval_results.json", "w") as f:
    json.dump(results, f, indent=2)


# ------------------- CELL 4: qualitative predictions -----------------
# Save side-by-side example detections: how the REAL-trained model (A)
# vs the MODEL-trained model (C) behave on the same test frames.
import shutil
test_imgs = sorted(glob.glob(os.path.join(DATASET_ROOT, "images", "test", "*")))[:8]
os.makedirs("/kaggle/working/qual", exist_ok=True)

for tag, wkey in [("real_trained_A", "A_real_640"), ("model_trained_C", "C_model_640")]:
    wp = WEIGHTS.get(wkey)
    if wp and os.path.exists(wp):
        mdl = YOLO(wp)
        mdl.predict(test_imgs, imgsz=IMGSZ[wkey], conf=0.25, save=True,
                    project="/kaggle/working/qual", name=tag, exist_ok=True)
print("Qualitative images saved under /kaggle/working/qual/")


# ------------------- CELL 5: zip everything --------------------------
shutil.make_archive("/kaggle/working/crosseval_export", "zip", "/kaggle/working/crosseval")
shutil.make_archive("/kaggle/working/qual_export", "zip", "/kaggle/working/qual")
print("\nDownload from Output tab:")
print("  crosseval_results.json  <-- the numbers for Table 5")
print("  crosseval_export.zip    <-- confusion matrices + PR curves on TEST")
print("  qual_export.zip         <-- qualitative detections (real-trained vs model-trained)")
