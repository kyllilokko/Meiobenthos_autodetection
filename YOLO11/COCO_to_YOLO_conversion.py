import os
import json
import glob
from PIL import Image

# Paths (Modify these as needed)
coco_annotations = {
    "train": "dataset/train/annotations/meiobenthos_train_8groups_augmented_coco.json",
    "val": "dataset/val/annotations/meiobenthos_val_8groups_coco.json"
}
image_dirs = {
    "train": "dataset/train/images",
    "val": "dataset/val/images"
}
output_dirs = {
    "train": "dataset/train/labels_8class",
    "val": "dataset/val/labels_8class"
}

# Ensure output label directories exist
for split in ["train", "val"]:
    os.makedirs(output_dirs[split], exist_ok=True)

# Function to convert COCO bbox to YOLO format
def coco_to_yolo_bbox(image_w, image_h, bbox):
    x_min, y_min, width, height = bbox
    x_center = (x_min + width / 2) / image_w
    y_center = (y_min + height / 2) / image_h
    width /= image_w
    height /= image_h
    return x_center, y_center, width, height

# Function to process COCO JSON and create YOLO labels
def convert_coco_to_yolo(split):
    with open(coco_annotations[split], "r") as f:
        coco_data = json.load(f)

    # Map category IDs to zero-based YOLO class IDs
    category_mapping = {cat["id"]: idx for idx, cat in enumerate(coco_data["categories"])}

    # Process each image
    image_id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}
    annotations_by_image = {}

    # Group annotations by image
    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    total_annotations = 0  # Track total annotations processed
    total_images_processed = 0  # Track number of images processed
    missing_files = 0  # Count missing image files

    # Convert annotations
    for image_id, filename in image_id_to_filename.items():
        # Handle file extensions correctly
        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(output_dirs[split], label_filename)
        image_path = os.path.join(image_dirs[split], filename)

        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Warning: Image file {image_path} not found!")
            missing_files += 1
            continue

        # Get image size
        with Image.open(image_path) as img:
            image_w, image_h = img.size

        # Debug: Print how many annotations exist for this image
        num_annotations = len(annotations_by_image.get(image_id, []))
        print(f"Processing {filename} -> {num_annotations} annotations")

        # Write YOLO annotations
        with open(label_path, "w") as label_file:
            if image_id in annotations_by_image:
                for ann in annotations_by_image[image_id]:
                    if ann["category_id"] not in category_mapping:
                        print(f"Warning: Category ID {ann['category_id']} missing in mapping, skipping annotation.")
                        continue

                    bbox = coco_to_yolo_bbox(image_w, image_h, ann["bbox"])
                    class_id = category_mapping[ann["category_id"]]
                    label_file.write(f"{class_id} {' '.join(map(str, bbox))}\n")
                    total_annotations += 1  # Count processed annotations

        total_images_processed += 1  # Count processed images

    # Debug: Count total label files generated
    label_files = glob.glob(f"{output_dirs[split]}/*.txt")
    print(f"\nConverted {split} set: {total_images_processed} images processed.")
    print(f"Total annotations processed: {total_annotations}")
    print(f"Total .txt label files created: {len(label_files)}")
    print(f"Total missing image files: {missing_files}\n")

# Run conversion for both train and val sets
convert_coco_to_yolo("train")
convert_coco_to_yolo("val")

print("✅ COCO to YOLO conversion complete!")
