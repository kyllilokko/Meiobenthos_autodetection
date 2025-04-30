# train models that detect meiofauna only (whatever groups, models just detect if an object is meiofauna)
import os
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torchvision.datasets import CocoDetection
import torchvision
import time
import datetime

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths
dataset_path = "dataset"
model_save_path = "models_trained-detect_1group"
log_path = os.path.join(model_save_path, f"training_log_{current_time}.txt")

# Define transformations
class CocoTransform:
    def __call__(self, image, target):
        image = F.to_tensor(image)  # Convert PIL image to tensor
        return image, target

# Dataset function
def get_coco_dataset(img_dir: str, ann_file: str):
    """Loads COCO dataset with custom transforms."""
    return CocoDetection(root=img_dir, annFile=ann_file, transforms=CocoTransform())

# Load datasets
train_dataset = get_coco_dataset(
    img_dir=os.path.join(dataset_path, "train"),
    ann_file=os.path.join(dataset_path, "train/annotations/meiobenthos_train_1class_250background_images_added_coco.json")
)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Model function. Load Faster R-CNN with ResNet-50 backbone
def get_model(num_classes):
    # Load pre-trained Faster R-CNN
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    # Replace the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Initialize the model
num_classes = 2  # Background + meiofauna
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model(num_classes).to(device)

# Define optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# Directory for saving models
os.makedirs(model_save_path, exist_ok=True)

# ---- Additional function to save checkpoints ----
def save_checkpoint(model, optimizer, scheduler, epoch, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")

# ---- Optional function to load checkpoint ----
def load_checkpoint(model, optimizer, scheduler, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded: {path} (Epoch {epoch})")
    return epoch

# Check for existing checkpoint to resume
latest_checkpoint = None
start_epoch = 0

# Automatically find the latest checkpoint
checkpoints = [f for f in os.listdir(model_save_path) if f.endswith('.pth')]
if checkpoints:
    # Sort by epoch number (assuming filename format: fasterrcnn_resnet50_epoch_{epoch}.pth)
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    latest_checkpoint = os.path.join(model_save_path, checkpoints[-1])

if latest_checkpoint:
    start_epoch = load_checkpoint(model, optimizer, lr_scheduler, latest_checkpoint, device) + 1
    print(f"Resuming training from epoch {start_epoch}")
else:
    print("Starting training from scratch")


# Training Function
def train_one_epoch(model, optimizer, data_loader, device, epoch, log_file):
    model.train()  # Set the model to train mode
    total_loss = 0.0

    for images, targets in data_loader:
        # Move images to the device
        images = [img.to(device) for img in images]

        # Validate and process targets
        processed_targets = []
        valid_images = []
        for i, target in enumerate(targets):
            boxes = []
            labels = []
            for obj in target:
                if "bbox" in obj:
                    x, y, w, h = obj["bbox"]
                    if w > 0 and h > 0:
                        boxes.append([x, y, x + w, y + h])
                        labels.append(obj["category_id"])

            if boxes:
                processed_targets.append({
                    "boxes": torch.tensor(boxes, dtype=torch.float32).to(device),
                    "labels": torch.tensor(labels, dtype=torch.int64).to(device),
                })
                valid_images.append(images[i])

        if not processed_targets:
            continue

        # Forward pass
        loss_dict = model(valid_images, processed_targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backpropagation
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Epoch [{epoch}] Loss: {avg_loss:.4f}")

    return avg_loss  # Return loss for logging


# Training loop with loss logging
with open(log_path, "a") as log_file:
    log_file.write(f"--- Training Started at {datetime.datetime.now()} ---\n")

    num_epochs = 55
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, log_file)
        lr_scheduler.step()

        # Save full checkpoint (model + optimizer + scheduler + epoch)
        checkpoint_path = os.path.join(model_save_path, f"fasterrcnn_resnet50_epoch_{epoch + 1}.pth")
        save_checkpoint(model, optimizer, lr_scheduler, epoch, checkpoint_path)

        # Log loss to file
        log_file.write(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}\n")
        log_file.write(f"{datetime.datetime.now()} - Checkpoint saved: {checkpoint_path}\n")

        end_time = time.time()
        print(f"Time per epoch: {(end_time - start_time) / 60:.2f} minutes")

    log_file.write(f"--- Training Completed at {datetime.datetime.now()} ---\n")