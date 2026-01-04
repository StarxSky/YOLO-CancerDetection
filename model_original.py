#%%writefile model.py
# Import PyTorch and related libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Helper libraries
import numpy as np
import pandas as pd
import pydicom
import os
import time
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

###             SET MODEL CONFIGURATIONS             ###
# Data Loading
CSV_PATH = 'label_data/CCC_clean.csv'
IMAGE_BASE_PATH = 'data/'
test_size_percent = 0.15

# Data Augmentation (simple for now; can extend with torchvision.transforms)
mirror_im = False

# Loss
lambda_coord = 5.0
epsilon = 1e-5

# Learning
learning_rate = 0.00001
BATCH_SIZE = 128
num_epochs = 60

# Saving
MODEL_SAVE_PATH = 'trained_model/model_pytorch.pth'

# TensorBoard
tb_log_dir = f'runs/yolo_cancer_{int(time.time())}'


###         CUSTOM DATASET CLASS         ###
class CancerDetectionDataset(Dataset):
    def __init__(self, img_paths, points, base_path=IMAGE_BASE_PATH, augment=False):
        self.img_paths = img_paths
        self.points = points  # Already normalized [0,1]
        self.base_path = base_path
        self.augment = augment

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = os.path.join(self.base_path, self.img_paths[idx])
        # Load DICOM
        image = pydicom.dcmread(path).pixel_array.astype(np.float32)

        # Normalize to [0, 1]
        image -= image.min()
        image_min = image.min()
        if image.max() - image_min > 0:
            image = (image - image_min) / (image.max() - image_min)
        
        # Add channel dimension: (512, 512) -> (1, 512, 512)
        image = np.expand_dims(image, axis=0)

        # To tensor
        image = torch.from_numpy(image)  # (1, 512, 512)

        # Target: normalized points [x1, y1, x2, y2] in [0,1]
        target = torch.tensor(self.points[idx], dtype=torch.float32)

        # Simple augmentation (horizontal flip)
        if self.augment and mirror_im and torch.rand(1) > 0.5:
            image = torch.flip(image, dims=[2])  # flip width
            # Adjust bounding box: x1 <-> x2
            target = target.clone()
            target[[0, 2]] = 1.0 - target[[2, 0]]

        return image, target


###            MODEL DEFINITION (YOLO-inspired)            ###
class YOLOCancerDetector(nn.Module):
    def __init__(self):
        super(YOLOCancerDetector, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),  # 512 -> 256
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                # 256 -> 128

            # Block 2
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                # 128 -> 64

            # Block 3
            nn.Conv2d(192, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                # 64 -> 32

            # Block 4 (repeated 1x1 -> 3x3 pattern)
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                # 32 -> 16

            # Block 5
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(inplace=True),

            # Final two convs (no pooling)
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Now spatial size is 8x8 (1024 channels)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 8 * 8, 1024),   # ← FIXED: 65536 → 1024
            nn.ReLU(inplace=True),
            nn.Linear(1024, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


###            CUSTOM LOSS AND METRIC FUNCTIONS            ###
def iou_metric(y_true, y_pred):
    """
    Compute mean IoU between predicted and ground truth boxes.
    Both: (B, 4) with order [x1, y1, x2, y2] normalized in [0,1]
    """
    # Ensure correct order: x1 < x2, y1 < y2
    x1_t = torch.min(y_true[:, 0], y_true[:, 2])
    y1_t = torch.min(y_true[:, 1], y_true[:, 3])
    x2_t = torch.max(y_true[:, 0], y_true[:, 2])
    y2_t = torch.max(y_true[:, 1], y_true[:, 3])

    x1_p = torch.min(y_pred[:, 0], y_pred[:, 2])
    y1_p = torch.min(y_pred[:, 1], y_pred[:, 3])
    x2_p = torch.max(y_pred[:, 0], y_pred[:, 2])
    y2_p = torch.max(y_pred[:, 1], y_pred[:, 3])

    # Intersection
    xi1 = torch.max(x1_t, x1_p)
    yi1 = torch.max(y1_t, y1_p)
    xi2 = torch.min(x2_t, x2_p)
    yi2 = torch.min(y2_t, y2_p)

    inter_w = torch.clamp(xi2 - xi1, min=0)
    inter_h = torch.clamp(yi2 - yi1, min=0)
    inter_area = inter_w * inter_h

    # Union
    box_t_area = (x2_t - x1_t) * (y2_t - y1_t)
    box_p_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = box_t_area + box_p_area - inter_area

    iou = inter_area / (union_area + 1e-6)
    return iou.mean()


def yolo_loss(y_pred, y_true, lambda_coord=5.0):
    """
    Custom YOLO-style loss focusing on center and size differences.
    """
    # Midpoints
    x_mid_p = y_pred[:, 0] + (y_pred[:, 2] - y_pred[:, 0]) / 2
    y_mid_p = y_pred[:, 1] + (y_pred[:, 3] - y_pred[:, 1]) / 2
    x_mid_t = y_true[:, 0] + (y_true[:, 2] - y_true[:, 0]) / 2
    y_mid_t = y_true[:, 1] + (y_true[:, 3] - y_true[:, 1]) / 2

    # Widths and heights (using sqrt to penalize small errors less)
    w_p = torch.sqrt(torch.abs(y_pred[:, 2] - y_pred[:, 0]) + 1e-6)
    h_p = torch.sqrt(torch.abs(y_pred[:, 3] - y_pred[:, 1]) + 1e-6)
    w_t = torch.sqrt(torch.abs(y_true[:, 2] - y_true[:, 0]) + 1e-6)
    h_t = torch.sqrt(torch.abs(y_true[:, 3] - y_true[:, 1]) + 1e-6)

    # Squared differences
    center_loss = (x_mid_p - x_mid_t)**2 + (y_mid_p - y_mid_t)**2
    size_loss = (w_p - w_t)**2 + (h_p - h_t)**2

    total_loss = lambda_coord * (center_loss + size_loss)
    return total_loss.mean()


###            DATA PREPARATION            ###
print("Loading and processing data...")
data_frame = pd.read_csv(CSV_PATH)

# Normalize points to [0,1]
im_dims = 512.0
points = []
for _, row in data_frame.iterrows():
    pt = [
        row['start_x'] / im_dims,
        row['start_y'] / im_dims,
        row['end_x'] / im_dims,
        row['end_y'] / im_dims
    ]
    points.append(pt)

points = np.array(points, dtype=np.float32)
img_paths = data_frame['imgPath'].values

# Train-test split
train_paths, test_paths, train_points, test_points = train_test_split(
    img_paths, points, test_size=test_size_percent, random_state=42
)

train_dataset = CancerDetectionDataset(train_paths, train_points, augment=True)
test_dataset = CancerDetectionDataset(test_paths, test_points, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
print("Data preprocessing complete\n")


###            MODEL, OPTIMIZER, TENSORBOARD            ###
model = YOLOCancerDetector().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
writer = SummaryWriter(log_dir=tb_log_dir)

print("Starting training...\n")

global_step = 0
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    epoch_iou = 0.0

    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = yolo_loss(outputs, targets, lambda_coord)
            loss.backward()
            optimizer.step()

            # Metrics
            with torch.no_grad():
                iou = iou_metric(targets, outputs)

            epoch_loss += loss.item()
            epoch_iou += iou.item()
            global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{iou.item():.4f}'
            })

            # TensorBoard logging
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/iou', iou.item(), global_step)

    avg_loss = epoch_loss / len(train_loader)
    avg_iou = epoch_iou / len(train_loader)
    print(f"\nEpoch {epoch+1} - Avg Loss: {avg_loss:.4f}, Avg IoU: {avg_iou:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            val_loss += yolo_loss(outputs, targets, lambda_coord).item()
            val_iou += iou_metric(targets, outputs).item()

    val_loss /= len(test_loader)
    val_iou /= len(test_loader)
    print(f"Validation - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}\n")

    writer.add_scalar('val/loss', val_loss, epoch)
    writer.add_scalar('val/iou', val_iou, epoch)

writer.close()

###                 SAVING THE MODEL                 ###
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

# Optional: save full model
# torch.save(model, 'trained_model/full_model_pytorch.pt')
