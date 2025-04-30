# # trained Faster R-CNN models evaluation - runs models on validation dataset and calculates metrics, including per class metrics
import torch
import torchvision
import os
import csv
import datetime
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from torchvision.datasets import CocoDetection
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define number of classes
num_classes = 10 # Background + nematode + rotifer + Testacea + ciliate + turbellarians + annelida + arthropoda + gastrotricha + tardigrada

# Function to load Faster R-CNN model
def get_model(num_classes):
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model


# Load validation dataset
class CocoTransform:
    def __call__(self, image, target):
        return F.to_tensor(image), target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


val_dataset = CocoDetection(
    root="dataset/val",
    annFile="dataset/val/annotations/meiobenthos_val_9groups_coco.json",
    transforms=CocoTransform()
)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)


# Function to compute IoU
def compute_iou(box1, boxes2):
    x1, y1, x2, y2 = box1
    ious = []
    for box in boxes2:
        x1g, y1g, w, h = box
        x2g, y2g = x1g + w, y1g + h
        xi1, yi1 = max(x1, x1g), max(y1, y1g)
        xi2, yi2 = min(x2, x2g), min(y2, y2g)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2g - x1g) * (y2g - y1g)
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area if union_area > 0 else 0
        ious.append(iou)
    return ious

# ➡️ Evaluate model with per-class metrics precision/recall/F1 etc
def evaluate_model(model, data_loader, device, num_classes, iou_threshold=0.5):
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i in range(len(images)):
                gt_labels = [obj["category_id"] for obj in targets[i]]
                gt_boxes = torch.tensor([obj["bbox"] for obj in targets[i]], dtype=torch.float32)
                pred_labels = outputs[i]["labels"].cpu().numpy()
                pred_scores = outputs[i]["scores"].cpu().numpy()
                pred_boxes = outputs[i]["boxes"].cpu().numpy()

                # Filter predictions (confidence > 0.5)
                keep = pred_scores > 0.5
                pred_labels, pred_boxes = pred_labels[keep], pred_boxes[keep]

                matched_preds, matched_labels = [], []
                assigned_gt = set()

                for pred_box, pred_label in zip(pred_boxes, pred_labels):
                    ious = compute_iou(pred_box, gt_boxes.numpy())
                    best_iou_idx = np.argmax(ious) if len(ious) > 0 else -1
                    best_iou = ious[best_iou_idx] if best_iou_idx >= 0 else 0

                    if best_iou >= iou_threshold and best_iou_idx not in assigned_gt:
                        matched_preds.append(pred_label)
                        matched_labels.append(gt_labels[best_iou_idx])
                        assigned_gt.add(best_iou_idx)
                    else:
                        matched_preds.append(pred_label)
                        matched_labels.append(0)  # False positive

                for idx, label in enumerate(gt_labels):
                    if idx not in assigned_gt:
                        matched_preds.append(0)  # Missed detection
                        matched_labels.append(label)

                all_preds.extend(matched_preds)
                all_labels.extend(matched_labels)

    # Compute overall metrics
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=1)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=1)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=1)
    accuracy = accuracy_score(all_labels, all_preds)

    # Compute per-class metrics
    per_class_metrics = {}
    for class_id in range(1, num_classes):
        class_precision = precision_score(all_labels, all_preds, labels=[class_id], average="micro", zero_division=1)
        class_recall = recall_score(all_labels, all_preds, labels=[class_id], average="micro", zero_division=1)
        class_f1 = f1_score(all_labels, all_preds, labels=[class_id], average="micro", zero_division=1)
        per_class_metrics[class_id] = (class_precision, class_recall, class_f1)

    return precision, recall, f1, accuracy, per_class_metrics

# ➡️ Evaluate model using COCO Evaluator
def evaluate_coco_map(model, data_loader, device, annFile):
    model.eval()

    coco_gt = COCO(annFile)
    coco_results = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for target, output in zip(targets, outputs):
                # Get image_id (assuming all annotations have the same image_id)
                image_id = target[0]["image_id"]

                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                # Convert boxes from [xmin, ymin, xmax, ymax] to [x, y, w, h]
                boxes_xywh = []
                for box in boxes:
                    x_min, y_min, x_max, y_max = box
                    width = x_max - x_min
                    height = y_max - y_min
                    boxes_xywh.append([x_min, y_min, width, height])

                # Append results in COCO format
                for box, score, label in zip(boxes_xywh, scores, labels):
                    coco_results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": box,
                        "score": float(score)
                    })

    # Load predictions to COCO
    coco_dt = coco_gt.loadRes(coco_results)

    # Evaluate mAP50 and mAP50-95
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP50 = coco_eval.stats[0]  # AP at IoU=0.50
    mAP50_95 = coco_eval.stats[1]  # AP averaged over IoU thresholds (0.50:0.95)

    return mAP50, mAP50_95


# Get list of models from folder
models_dir = "9groups_models_trained"
model_files = [f for f in os.listdir(models_dir) if f.endswith(".pth")]

# ➡️ CSV file setup
csv_filename = os.path.join(models_dir, "evaluation_results.csv")
if not os.path.exists(csv_filename):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ["Model Name", "Precision", "Recall", "F1-score", "Accuracy", "mAP@50", "mAP@[50:95]"]
        for class_id in range(1, num_classes):  # Class 1 to 9
            header.extend([f"Class {class_id} Precision", f"Class {class_id} Recall", f"Class {class_id} F1-score"])
        header.append("Evaluation Date")
        writer.writerow(header)

    # ➡️ Evaluate each model
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        print(f"\n🔍 Evaluating {model_file}...")

        annFile="dataset/val/annotations/meiobenthos_val_9groups_coco.json"

        model = get_model(num_classes=num_classes)
        # Load the checkpoint file
        checkpoint = torch.load(model_path, map_location=device)
        # Load only the model state dict from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        # ➡️ Precision / Recall / F1 Evaluation
        precision, recall, f1, accuracy, per_class_metrics = evaluate_model(model, val_loader, device, num_classes)

        # ➡️ COCO mAP Evaluation
        mAP_50_95, mAP_50 = evaluate_coco_map(model, val_loader, device, annFile)

        eval_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Build row data
        row = [model_file, precision, recall, f1, accuracy, mAP_50, mAP_50_95]
        for class_id in range(1, num_classes):
            row.extend(per_class_metrics[class_id])
        row.append(eval_date)

        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

        print(f"✅ Results for {model_file}:")
        print(f"   Precision = {precision:.4f}")
        print(f"   Recall    = {recall:.4f}")
        print(f"   F1-Score  = {f1:.4f}")
        print(f"   Accuracy  = {accuracy:.4f}")
        print(f"   mAP@50    = {mAP_50:.4f}")
        print(f"   mAP@50-95 = {mAP_50_95:.4f}")
        #print(f"\nClassification Report:\n{class_report}")

        # Print per-class metrics
        print("\nPer-Class Metrics:")
        for class_id in range(1, num_classes):
            class_precision, class_recall, class_f1 = per_class_metrics[class_id]
            print(f"   Class {class_id}: Precision={class_precision:.4f}, Recall={class_recall:.4f}, F1={class_f1:.4f}")

print(f"\n🎉 All results saved to {csv_filename}")