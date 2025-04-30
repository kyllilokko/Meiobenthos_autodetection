# trains YOLO11 models and calculates metrics (averages only, not per class)
from ultralytics import YOLO
import datetime
import time
import shutil
import os
import pandas as pd  # Import for analyzing results.csv


# 🕒 Get current timestamp for logging
current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = f"training_log_{current_date}.txt"

# 🎯 Load YOLO model
#model = YOLO("runs/detect/train_8class2025-04-05_15-06-51/weights/last.pt") # provide last trained model to continue if training was interrupted
model = YOLO("yolo11m.pt") # to train from scratch

# 🚀 Training parameters
epochs = 90  # Adjust as needed
batch_size = 4
img_size = 1024
data_yaml = "dataset_1group_only/data.yaml"  # Ensure this file exists and is correctly configured, data.yaml contains paths that needs to be updated

# YOLO save directory
project_dir = os.path.join("runs", "detect")
train_folder_name = f"train_1groups{current_date}"

# 📝 Open log file for writing
with open(log_path, "a") as log_file:
    log_file.write(f"--- Training Started at {datetime.datetime.now()} ---\n")

    start_time = time.time()  # Start timer

    # 🔥 Train YOLO model
    results = model.train(
        # resume=True, # comment out if not resuming training (if starting from beginning)
        data=data_yaml,
        epochs=epochs,  # Train for all epochs at once
        batch=batch_size,
        imgsz=img_size,
        device= "cuda" if model.device.type == "cuda" else "cpu",
        verbose=True,
        project=project_dir,
        name = train_folder_name
    )

    # ⏳ Compute training time
    end_time = time.time()
    log_file.write(f"Total Training Time: {(end_time - start_time) / 60:.2f} minutes\n")

    # ✅ Save best model (from YOLO's default save path)
    best_model_src = f"{project_dir}/{train_folder_name}/weights/best.pt"
    best_model_dest = f"best_model_{current_date}.pt"

    if os.path.exists(best_model_src):
        shutil.copy(best_model_src, best_model_dest)
        log_file.write(f"✅ Best model copied to: {best_model_dest}\n")
        print(f"✅ Best model copied to: {best_model_dest}")
    else:
        log_file.write("⚠️ Best model not found! Check training folder.\n")
        print("⚠️ Best model not found in runs/detect/train/weights/.")

# 🏆 Identify Best Model Epoch from results.csv
results_csv_path = f"{project_dir}/{train_folder_name}/results.csv"

if os.path.exists(results_csv_path):
    df = pd.read_csv(results_csv_path)

    # Find the epoch with the highest mAP@50 (best model criteria)
    best_epoch_index = df["metrics/mAP50-95(B)"].idxmax()
    best_epoch = df.loc[best_epoch_index]

    best_epoch_num = int(best_epoch["epoch"])  # Get epoch number
    best_map50 = best_epoch["metrics/mAP50(B)"]
    best_map50_95 = best_epoch["metrics/mAP50-95(B)"]
    best_precision = best_epoch["metrics/precision(B)"]
    best_recall = best_epoch["metrics/recall(B)"]

    # Compute F1-score for the best model
    best_f1_score = (2 * best_precision * best_recall) / (best_precision + best_recall) if (best_precision + best_recall) > 0 else 0

    # 📝 Log Best Model Metrics
    with open(log_path, "a") as log_file:
        log_file.write("\n--- Best Model Identified ---\n")
        log_file.write(f"Best Model Epoch: {best_epoch_num}\n")
        log_file.write(f"mAP@50: {best_map50:.4f}\n")
        log_file.write(f"mAP@50-95: {best_map50_95:.4f}\n")
        log_file.write(f"Precision: {best_precision:.4f}\n")
        log_file.write(f"Recall: {best_recall:.4f}\n")
        log_file.write(f"F1-score: {best_f1_score:.4f}\n")

    print(f"✅ Best model was from epoch {best_epoch_num} with mAP@50-95: {best_map50_95:.4f}")

else:
    print("⚠️ Results CSV not found! Could not determine best epoch.")

print(f"✅ Training completed! Logs saved to {log_path}")