import os
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm
import pandas as pd
import datetime

# Global defaults
TEST_IMAGE_FOLDER = "test_images"
DEFAULT_MODEL1_PATH = "models_for_test/fasterrcnn_resnet50_epoch_28_1group-detection_only.pth"
#DEFAULT_MODEL2_PATH = "models_for_test/fasterrcnn_resnet50_epoch_11_9groups.pth"
DEFAULT_MODEL2_PATH = "models_for_test/fasterrcnn_resnet50_epoch_22_8groups.pth"

current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H")

# Define class labels (update according to your dataset)
CLASS_NAMES = ["background", "meiofauna"]

# 9 groups
#CLASS_NAMES_SECOND = [ # 9 groups
#    "background", "nematode", "rotifer", "Testacea", "ciliate",
#    "turbellarians", "annelida", "arthropoda", "gastrotricha",
#    "tardigrada"
#]

# 8 groups
CLASS_NAMES_SECOND = [
    "background", "nematode", "rotifer", "Testacea", "ciliate",
    "turbellarians", "annelida", "arthropoda",
    "tardigrada"
]

def load_model(model_path, num_classes, device):
    """
    Loads the Faster R-CNN model with a custom predictor and trained weights.
    """
    model = fasterrcnn_resnet50_fpn(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)

    # If the checkpoint is a dictionary with "model_state_dict"
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def draw_boxes(image: Image.Image, boxes, labels, scores, threshold: float, class_names) -> Image.Image:
    """
    Draws ALL bounding boxes and labels on the provided image,
    if their confidence scores are above the threshold.
    """
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=35)
    except IOError:
        font = ImageFont.load_default()

    # Filter indices with scores above threshold
    indices = np.where(scores >= threshold)[0]
    if len(indices) == 0:
        return image  # No predictions above threshold

    for box, label, score in zip(boxes, labels, scores):
        if score >= threshold:
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            text = f"{class_names[label]}: {score:.2f}"
            text_bbox = draw.textbbox((x1, y1), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_background = [x1, y1, x1 + text_width, y1 + text_height]
            draw.rectangle(text_background, fill="red")
            draw.text((x1, y1), text, fill="yellow", font=font)

    return image


def process_images(folder, results_folder, model, class_names, device, threshold=0.5):
    os.makedirs(results_folder, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor()])
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    report_data = {}

    for file in tqdm(files, desc=f"Processing {results_folder}"):
        image_path = os.path.join(folder, file)
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(image_tensor)

        boxes = prediction[0]["boxes"].cpu().numpy()
        labels = prediction[0]["labels"].cpu().numpy()
        scores = prediction[0]["scores"].cpu().numpy()

        # Save annotated image
        annotated_image = draw_boxes(image.copy(), boxes, labels, scores, threshold, class_names)
        annotated_path = os.path.join(results_folder, f"annotated_{file}")
        annotated_image.save(annotated_path)

        # Report details
        indices = np.where(scores >= threshold)[0]
        detected_organisms = []
        scores_above_threshold = []

        for idx in indices:
            label_index = labels[idx]
            if label_index < len(class_names):
                detected_organisms.append(class_names[label_index])
            else:
                detected_organisms.append(f"Unknown({label_index})")
            scores_above_threshold.append(scores[idx])

        alert_value = ""
        if scores_above_threshold:
            min_score = min(scores_above_threshold)
            if min_score < 0.75:
                alert_value = f"{min_score:.2f}"

        report_data[file] = {
            "Detected Organisms": ", ".join(detected_organisms) if detected_organisms else "None",
            "Alert (Score < 0.75)": alert_value
        }

    return report_data


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = load_model(DEFAULT_MODEL1_PATH, len(CLASS_NAMES), device)
    model2 = load_model(DEFAULT_MODEL2_PATH, len(CLASS_NAMES_SECOND), device)

    results_model1 = process_images(TEST_IMAGE_FOLDER, f"results_model1_{current_date}", model1, CLASS_NAMES, device)
    results_model2 = process_images(TEST_IMAGE_FOLDER, f"results_model2_{current_date}", model2, CLASS_NAMES_SECOND, device)

    all_files = sorted(set(results_model1.keys()) | set(results_model2.keys()))
    combined_rows = []

    for file in all_files:
        row = {
            "Image File Name": file,
            "Model 1: Detected Organisms": results_model1.get(file, {}).get("Detected Organisms", "None"),
            "Model 1: Alert (Score < 0.75)": results_model1.get(file, {}).get("Alert (Score < 0.75)", ""),
            "Model 2: Detected Organisms": results_model2.get(file, {}).get("Detected Organisms", "None"),
            "Model 2: Alert (Score < 0.75)": results_model2.get(file, {}).get("Alert (Score < 0.75)", "")
        }
        combined_rows.append(row)

    final_report_df = pd.DataFrame(combined_rows)
    os.makedirs(f"final_results_{current_date}", exist_ok=True)
    report_path = os.path.join("final_results", f"final_combined_report_{current_date}.xlsx")
    final_report_df.to_excel(report_path, index=False)
    print(f"✅ Final combined report saved to {report_path}")


if __name__ == "__main__":
    main()
