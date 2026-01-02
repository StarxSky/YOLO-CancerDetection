#%%writefile predict.py
# Import PyTorch
import torch
import torch.nn as nn
from torchvision import transforms

# Imports for visualizing predictions
import numpy as np
import pydicom
from skimage.transform import resize
import PIL
from PIL import Image, ImageDraw, ImageColor

# Helper imports
import sys, os
import pandas as pd
import time

# Path variables - same as in training script (bb.py)
weights_path = 'trained_model/model_weights.pth'  # Model saved in bb.py
CSV_PATH = 'label_data/CCC_clean.csv'
IMAGE_BASE_PATH = 'data/'

# Global variables
img_dims = 512

# Define the YOLOModel class exactly as in training script (bb.py)
class YOLOModel(nn.Module):
    def __init__(self):
        super(YOLOModel, self).__init__()
        
        # Define the convolutional layers
        self.conv_layers = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 2
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 3
            nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 4 (repeat block 4 times)
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            # Layer 5
            nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 6
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # Layer 7
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Define the fully connected layers with the corrected input dimension
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 8 * 8, 1024),  # Corrected input dimension from 4*4 to 8*8
            nn.Linear(1024, 4096),
            nn.Linear(4096, 4),
            nn.Sigmoid()  # 4 outputs: predict 4 points for a bounding box
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def load_model(weights_file):
    """
    Load a PyTorch model from a .pth file found at provided path.

    @param weights_file - path to valid .pth file of YOLO cancer detection model
    @return model - a fully trained PyTorch model
    """
    print(f"Loading model from {weights_file}...")
    
    # Create model instance
    model = YOLOModel()
    
    # Load weights into model
    try:
        model.load_state_dict(torch.load(weights_file, map_location=torch.device('cpu')))
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"Error: Model weights file not found at {weights_file}")
        print("Please ensure you have trained the model first using bb.py")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Set model to evaluation mode
    model.eval()
    
    return model

def is_dicom(im_path):
    """
    Check if the image specified by the path is a DICOM image.

    @param im_path - string path to an image file
    @return - boolean, True if is a dicom image, else False
    """
    # get file extension
    path, ext = os.path.splitext(im_path)
    
    # if file extension is empty or is .dcm, assume DICOM file
    if ext == ".dcm" or ext == "":
        return True
    else:
        return False

def load_image(im_path):
    """
    Load an image from provided path. Loads both DICOM and more
    common image formats into a numpy array.

    @param im_path - string path to the image file
    @return im - the image loaded as a numpy ndarray
    """
    try:
        if is_dicom(im_path):
            # load with pydicom
            im = pydicom.dcmread(im_path).pixel_array
            return im
        else:
            # load with Pillow
            im = Image.open(im_path)
            return np.array(im)
    except Exception as e:
        print(f"Error loading image {im_path}: {e}")
        sys.exit(1)

def pre_process(img):
    """
    Takes an image and preprocesses it to fit in the model

    @param img - a numpy ndarray representing an image
    @return - a shaped and normalized, grayscale version of img as PyTorch tensor
    """
    # resize image to 512x512
    im_adjusted = resize(img, (img_dims, img_dims), anti_aliasing=True,
                             preserve_range=True)

    # ensure image is grayscale (only has 1 channel)
    im_adjusted = im_adjusted.astype(np.float32)
    if len(im_adjusted.shape) >= 3:
        # squash 3 channel image to grayscale
        im_adjusted = np.dot(im_adjusted[...,:3], [0.299, 0.587, 0.114])
    
    # normalize the image to a 0-1 range
    if not np.amax(im_adjusted) < 1: # check that image isn't already normalized
        if np.amin(im_adjusted) < 0:
            im_adjusted += np.amin(im_adjusted)
        im_adjusted /= np.amax(im_adjusted)
    
    # Convert to PyTorch tensor and add batch dimension and channel dimension
    im_adjusted = torch.from_numpy(im_adjusted).unsqueeze(0).unsqueeze(0)
    
    return im_adjusted

def normalize_image(img):
    """
    Normalize an image to the range of 0-255. This may help reduce the white
    washing that occurs with displaying DICOM images with PIL.

    @param img - a numpy array representing an image
    @return normalized - a numpy array whose elements are all within the range 
                         of 0-255
    """
    # adjust for negative values
    normalized = img + np.abs(np.amin(img))
    
    # normalize to 0-1
    normalized = normalized.astype(np.float32)
    normalized /= np.amax(normalized)
    
    # stretch scale of range to 255
    normalized *= 255
    return normalized

def predict_image(model, img_path, save_output=False, output_path='prediction_result.png'):
    """
    Make a prediction on a single image using the trained model.
    
    @param model - trained PyTorch model
    @param img_path - path to input image
    @param save_output - whether to save the prediction result as an image
    @param output_path - path to save the prediction result
    @return prediction - the predicted bounding box coordinates
    """
    # Load and preprocess the image
    img = load_image(img_path)
    preprocessed_img = pre_process(img)
    
    # Make prediction with PyTorch model
    with torch.no_grad():
        output = model(preprocessed_img)
    
    # Un-normalize prediction to get plotable points
    prediction = np.array(output[0].cpu()) * 512  # Move tensor to CPU before converting to numpy
    prediction = prediction.astype(np.int32)
    
    # Ensure proper order: [x0, y0, x1, y1] where x0 <= x1 and y0 <= y1
    x0, y0, x1, y1 = prediction
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    prediction = [x0, y0, x1, y1]
    
    print(f"Predicted bounding box: {prediction}")
    
    # Visualize the result if save_output is True
    if save_output:
        # Normalize image to prevent as much white-washing caused by PIL lib as possible
        norm = normalize_image(img)
        
        # Draw bbox of predicted points
        im = Image.fromarray(norm).convert("RGB")  # Convert RGB for colored bboxes
        draw = ImageDraw.Draw(im)
        draw.rectangle(prediction, outline='#ff0000', width=2)  # Red bbox with width=2
        
        # Save the result
        im.save(output_path)
        print(f"Prediction result saved to {output_path}")
    
    return prediction

def predict_from_csv(model, csv_path, image_base_path, num_images=5, save_outputs=False, output_dir='predictions'):
    """
    Make predictions on images listed in a CSV file using the trained model.
    
    @param model - trained PyTorch model
    @param csv_path - path to CSV file with image paths and ground truth labels
    @param image_base_path - base path for image files
    @param num_images - number of images to predict on
    @param save_outputs - whether to save prediction results as images
    @param output_dir - directory to save prediction results
    """
    # Load the CSV file
    try:
        data_frame = pd.read_csv(csv_path)
        print(f"Loaded {len(data_frame)} images from {csv_path}")
    except Exception as e:
        print(f"Error loading CSV file {csv_path}: {e}")
        sys.exit(1)
    
    # Create output directory if needed
    if save_outputs and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Make predictions for the specified number of images
    for i in range(min(num_images, len(data_frame))):
        img_path = image_base_path + data_frame['imgPath'][i]
        print(f"\nProcessing image {i+1}/{min(num_images, len(data_frame))}: {img_path}")
        
        # Get ground truth points for comparison
        true_points = [int(data_frame['start_x'][i]),
                      int(data_frame['start_y'][i]),
                      int(data_frame['end_x'][i]),
                      int(data_frame['end_y'][i])]
        
        # Ensure proper order for ground truth points
        x0, y0, x1, y1 = true_points
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        true_points = [x0, y0, x1, y1]
        
        print(f"Ground truth bounding box: {true_points}")
        
        # Make prediction
        prediction = predict_image(model, img_path, save_output=False)
        
        # Visualize and save if requested
        if save_outputs:
            # Load and normalize the image
            img = load_image(img_path)
            norm = normalize_image(img)
            
            # Draw bboxes
            im = Image.fromarray(norm).convert("RGB")
            draw = ImageDraw.Draw(im)
            draw.rectangle(prediction, outline='#ff0000', width=2)  # Red predicted bbox
            draw.rectangle(true_points, outline='#00ff00', width=2)  # Green ground truth bbox
            
            # Save the result
            output_path = os.path.join(output_dir, f"prediction_{i+1}.png")
            im.save(output_path)
            print(f"Prediction result saved to {output_path}")

def main():
    """
    Main function to run the inference script.
    """
    print("=== YOLO Cancer Detection Inference ===")
    
    # Load the trained model
    model = load_model(weights_path)
    
    # Option 1: Predict on all images in CSV file (first 5 images)
    print("\n=== Predicting from CSV file ===")
    predict_from_csv(model, CSV_PATH, IMAGE_BASE_PATH, num_images=5, save_outputs=True)
    
    # Option 2: Predict on a single image (uncomment and modify the path)
    # print("\n=== Predicting on single image ===")
    # img_path = "data/example_image.dcm"
    # predict_image(model, img_path, save_output=True, output_path="single_prediction.png")
    
    print("\n=== Inference completed ===")

if __name__ == '__main__':
    main()
