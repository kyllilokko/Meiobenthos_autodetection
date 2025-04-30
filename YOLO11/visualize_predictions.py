import os
import torch
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from tqdm import tqdm
import datetime

current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H")

# Global defaults
MODEL_1_PATH = "models_for_test/best_YOLO11_model1_1group_detection_only.pt"
MODEL_2_PATH = "models_for_test/best_YOLO11_model2_8groups.pt"
#MODEL_2_PATH = "models_for_test/best_YOLO11_model2_9groups.pt"
TEST_IMAGE_FOLDER = "test_images"
RESULTS_FOLDER_1 = f"results_model1_{current_date}"
RESULTS_FOLDER_2 = f"results_model2_{current_date}"
REPORT_FILE_1 = f"report_model1_{current_date}.xlsx"
REPORT_FILE_2 = f"report_model2_{current_date}.xlsx"

CLASS_NAMES_1 = ["meiofauna"]
# 9 groups
#CLASS_NAMES_2 = [
#    "nematode", "rotifer", "Testacea", "ciliate",
#    "turbellarians", "annelida", "arthropoda", "gastrotricha", "tardigrada"
#]
# 8 groups
CLASS_NAMES_2 = [
    "nematode", "rotifer", "Testacea", "ciliate",
    "turbellarians", "annelida", "arthropoda", "tardigrada"
]

# Load YOLO model
def load_model(model_path: str, device: torch.device):
    model = YOLO(model_path)
    model.to(device)
    return model


# Draw bounding boxes
def draw_boxes(image: Image.Image, boxes, labels, scores, threshold: float, class_names) -> Image.Image:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", size=35)
    except IOError:
        font = ImageFont.load_default()

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


# process images with models and return report data
def process_yolo_model(images, model, device, class_names, result_folder, threshold=0.5):
    os.makedirs(result_folder, exist_ok=True)
    report_data = {}
    for img_name in tqdm(images, desc=f"Processing {result_folder}"):
        image = Image.open(os.path.join(TEST_IMAGE_FOLDER, img_name)).convert("RGB")
        preds = model(image)[0].boxes.data.cpu().numpy()
        boxes = preds[:, :4]
        scores = preds[:, 4]
        labels = preds[:, 5].astype(int)
        image_with_boxes = draw_boxes(image.copy(), boxes, labels, scores, threshold, class_names)
        image_with_boxes.save(os.path.join(result_folder, f"annotated_{img_name}"))

        all_classes = []
        alert_values = []
        for score, label in zip(scores, labels):
            if score >= threshold:  # Filter out predictions below threshold
                if label < len(class_names):
                    all_classes.append(class_names[label])
                else:
                    all_classes.append(f"Unknown({label})")
                if score < 0.75:
                    alert_values.append(f"{score:.2f}")

        report_data[img_name] = {
            "Classes": ", ".join(all_classes) if all_classes else "None",
            "Alert": ", ".join(alert_values) if alert_values else ""
        }
    return report_data


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = load_model(MODEL_1_PATH, device)
    model2 = load_model(MODEL_2_PATH, device)

    images = [f for f in os.listdir(TEST_IMAGE_FOLDER) if f.endswith(('.jpg', '.JPG', '.png'))]

    data1 = process_yolo_model(images, model1, device, CLASS_NAMES_1, RESULTS_FOLDER_1)
    data2 = process_yolo_model(images, model2, device, CLASS_NAMES_2, RESULTS_FOLDER_2)

    all_files = sorted(set(data1.keys()) | set(data2.keys()))
    combined = []
    for file in all_files:
        row = {
            "Image Name": file,
            "Model 1: Classes": data1.get(file, {}).get("Classes", "None"),
            "Model 1: Alert": data1.get(file, {}).get("Alert", ""),
            "Model 2: Classes": data2.get(file, {}).get("Classes", "None"),
            "Model 2: Alert": data2.get(file, {}).get("Alert", "")
        }
        combined.append(row)

    df_combined = pd.DataFrame(combined)
    os.makedirs("final_results", exist_ok=True)
    report_path = os.path.join("final_results", f"final_combined_yolo_report_{current_date}.xlsx")
    df_combined.to_excel(report_path, index=False)
    print(f"✅ Final YOLO combined report saved to {report_path}")



if __name__ == "__main__":
    main()