# Import PyTorch
import torch
import torch.nn as nn

# Imports for visualizing predictions
import numpy as np
import pydicom
from skimage.transform import resize
from PIL import Image, ImageDraw

# Helper imports
import sys
import os
import pandas as pd
import time

# Path variables 
MODEL_PATH = 'trained_model/model_pytorch.pth'  # Changed to PyTorch format
CSV_PATH = 'label_data/CCC_clean.csv'
IMAGE_BASE_PATH = 'data/'

# Global variables
img_dims = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -----------------------------
# Define your PyTorch model here
# -----------------------------
# Example: Simple CNN that outputs 4 bounding box coordinates (normalized)
class YOLOCancerDetector(nn.Module):
    def __init__(self):
        super(YOLOCancerDetector, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(192, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4 - repeated 1x1 and 3x3 convolutions
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
            nn.MaxPool2d(kernel_size=2, stride=2),

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
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # Final two convolutions
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Final feature map: 8x8 x 1024 → flattened = 65536
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 8 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4),
            nn.Sigmoid()  # Output normalized [0,1] bounding box coordinates
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_model(model_path):
    """
    Load a PyTorch model from a .pth file.

    @param model_path - path to .pth file containing state_dict
    @return model - loaded PyTorch model in evaluation mode
    """
    print("Loading PyTorch model from disk...", end=" ")
    
    # Instantiate the model (you must use the correct architecture)
    model = YOLOCancerDetector()  # <-- Replace with your actual model class if different
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # important: set to evaluation mode (no dropout, batchnorm fixed)
    
    print("Complete!")
    return model


def is_dicom(im_path):
    """Check if the image is a DICOM file."""
    _, ext = os.path.splitext(im_path)
    return ext.lower() in {".dcm", ""}  # empty extension often used for DICOM


def load_image(im_path):
    """Load image as numpy array (supports DICOM and standard formats)."""
    if is_dicom(im_path):
        return pydicom.dcmread(im_path).pixel_array
    else:
        return np.array(Image.open(im_path))


def pre_process(img):
    """
    Preprocess image to match model expectations:
    - Resize to 512x512
    - Convert to grayscale
    - Normalize to [0,1]
    - Add batch and channel dimensions: (1, 1, 512, 512)
    """
    # Resize
    im_adjusted = resize(img, (img_dims, img_dims), anti_aliasing=True, preserve_range=True)
    im_adjusted = im_adjusted.astype(np.float32)

    # Convert to grayscale if needed
    if len(im_adjusted.shape) >= 3:
        im_adjusted = np.dot(im_adjusted[..., :3], [0.299, 0.587, 0.114])

    # Normalize to [0, 1]
    if np.amax(im_adjusted) > 1.0:
        if np.amin(im_adjusted) < 0:
            im_adjusted -= np.amin(im_adjusted)
        im_adjusted /= np.amax(im_adjusted)

    # Add batch and channel dims: (1, 1, H, W)
    im_adjusted = im_adjusted[np.newaxis, np.newaxis, ...]  # shape: (1, 1, 512, 512)
    return torch.from_numpy(im_adjusted).to(device)


def normalize_image(img):
    """Normalize image to 0-255 range for better PIL display."""
    normalized = img - np.amin(img)
    normalized = normalized / np.amax(normalized)
    normalized = (normalized * 255).astype(np.uint8)
    return normalized


def main():
    """Load model, iterate through dataset, predict bounding boxes, and display results."""
    # Load the PyTorch model
    model = load_model(MODEL_PATH)

    # Load dataset CSV
    data_frame = pd.read_csv(CSV_PATH)

    print(f"Processing {len(data_frame)} images...")

    with torch.no_grad():
        output = model(input_tensor)  # (1, 4)

        # Predicted box
        pred = output.cpu().numpy()[0] * img_dims
        x1, y1, x2, y2 = pred
        x0 = int(max(0, min(x1, x2)))
        y0 = int(max(0, min(y1, y2)))
        x1 = int(min(img_dims - 1, max(x1, x2)))
        y1 = int(min(img_dims - 1, max(y1, y2)))
        pred_bbox = [x0, y0, x1, y1]
        # Ground truth box (safe)
        gt_x1 = int(data_frame['start_x'][i])
        gt_y1 = int(data_frame['start_y'][i])
        gt_x2 = int(data_frame['end_x'][i])
        gt_y2 = int(data_frame['end_y'][i])
        true_bbox = [
            max(0, min(gt_x1, gt_x2)),
            max(0, min(gt_y1, gt_y2)),
            min(img_dims - 1, max(gt_x1, gt_x2)),
            min(img_dims - 1, max(gt_y1, gt_y2))
        ]

        # Draw
        draw.rectangle(pred_bbox, outline="#ff0000", width=4)
        draw.rectangle(true_bbox, outline="#00ff00", width=4)
        # Labels
        draw.text((pred_bbox[0], pred_bbox[1] - 20), "Predicted", fill="#ff0000")
        draw.text((true_bbox[0], true_bbox[1] - 20), "Ground Truth", fill="#00ff00")
    
    

        pil_img.show()
        print(f"Displayed image {i+1}/{len(data_frame)}: {data_frame['imgPath'][i]}")
        time.sleep(1)  # Give user time to view/close

        # Optional: break early for testing
        # if i >= 9: break


if __name__ == '__main__':
    main()
