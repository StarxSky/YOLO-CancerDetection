#%%writefile model.py
# Import PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Helper libraries
import math
import numpy as np
import pandas as pd
import pydicom
import os
import sys
import time

# Imports for dataset manipulation
from sklearn.model_selection import train_test_split

# Improve progress bar display
import tqdm
import tqdm.auto
tqdm.tqdm = tqdm.auto.tqdm

#tf.enable_eager_execution() #comment this out if causing errors
#tf.logging.set_verbosity(tf.logging.DEBUG)



###             ARGUMENT PARSER CONFIGURATIONS             ###
def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Cancer Detection Training Script")
    
    # 数据相关路径
    parser.add_argument('--csv_path', type=str, default='label_data/CCC_clean.csv', 
                       help='Path to labels CSV')
    parser.add_argument('--image_path', type=str, default='data/', 
                       help='Base path for images')
    parser.add_argument('--weight_path', type=str, default='trained_model/model_weights.pth', 
                       help='Path to save model weights')
    
    # 训练超参数
    parser.add_argument('--epochs', type=int, default=1, 
                       help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=5, 
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.00001, 
                       help='Learning rate')
    parser.add_argument('--test_size', type=float, default=0.15, 
                       help='Validation split size (0.0-1.0)')
    
    # 模型特定参数
    parser.add_argument('--lambda_coord', type=float, default=5.0, 
                       help='YOLO loss coordinator weight')
    parser.add_argument('--epsilon', type=float, default=0.00001, 
                       help='Epsilon value for numerical stability')
    parser.add_argument('--mirror', action='store_true', 
                       help='Enable horizontal flip augmentation')
    parser.add_argument('--no_mirror', dest='mirror', action='store_false',
                       help='Disable horizontal flip augmentation')
    parser.set_defaults(mirror=False)
    
    # 其他选项
    parser.add_argument('--save_model', action='store_true', 
                       help='Save model after training')
    parser.add_argument('--no_save', dest='save_model', action='store_false',
                       help='Do not save model after training')
    parser.set_defaults(save_model=True)
    
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose output')
    
    return parser.parse_args()

# Device configuration
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else :
    device = torch.device('cpu')
    

            
###         DATASET AND DATA LOADING FUNCTIONS        ###

class DICOMDataset(Dataset):
    def __init__(self, img_paths, points, image_base_path, transform=None):
        self.img_paths = img_paths
        self.points = points
        self.image_base_path = image_base_path
        self.transform = transform
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.image_base_path, self.img_paths[idx])
        img = pydicom.dcmread(img_path).pixel_array
        
        # Normalize image
        img = img.astype(np.float32)
        img += abs(np.amin(img))  # account for negatives
        img /= np.amax(img)
        
        # Get points
        point = self.points[idx]
        
        # Convert to PyTorch tensors
        img = torch.from_numpy(img).unsqueeze(0)  # Add channel dimension
        point = torch.from_numpy(point).float()
        
        # Apply transformations if any
        if self.transform:
            img = self.transform(img)
        
        return img, point

def normalize_points(points, imDims=512.0):
    """
    Normalize values in points to be within the range of 0 to 1.
    
    @param points - list of tuples, each element valued in the range of 0
                    512 (inclusive). This is known from the nature
                    of the dataset used in this program
    @param imDims - image dimensions (default 512x512)
    @return - list of numpy ndarrays (float), elements valued in range
              0 to 1 (inclusive)
    """
    normalized_points = []
    from tqdm import tqdm
    
    for point in tqdm(list(points)):
        normalized_point = np.array(point) / imDims
        normalized_points.append(normalized_point.astype(np.float32))
    
    return normalized_points

def load_and_prepare_data(csv_path, image_base_path, test_size=0.15, mirror=False, batch_size=5):
    """
    Load and prepare training and testing datasets.
    
    @param csv_path - path to the CSV file containing labels
    @param image_base_path - base path for images
    @param test_size - fraction of data to use for testing
    @param mirror - whether to apply horizontal flip augmentation
    @param batch_size - batch size for data loaders
    @return - train_loader, test_loader, num_train_examples, num_test_examples
    """
    print("Loading and processing data\n") 
    
    data_frame = pd.read_csv(csv_path)
    
    # zip all points for each image label together into a tuple 
    points = zip(data_frame['start_x'], data_frame['start_y'], \
                           data_frame['end_x'], data_frame['end_y'])
    img_paths = data_frame['imgPath']
    
    # Prepare transformations
    transform_list = []
    if mirror:
        transform_list.append(transforms.RandomHorizontalFlip(p=1.0))
    
    transform = transforms.Compose(transform_list)
    
    # Normalize points
    points = normalize_points(points)
    
    # Convert to numpy arrays
    img_paths = np.array(img_paths)
    points = np.array(points)
    
    # split the data into train and test
    train_img_paths, test_img_paths, train_points, test_points = \
        train_test_split(img_paths, points, test_size=test_size, random_state=42)
    
    # Create datasets
    train_dataset = DICOMDataset(train_img_paths, train_points, image_base_path, transform=transform)
    test_dataset = DICOMDataset(test_img_paths, test_points, image_base_path)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    num_train_examples = len(train_dataset)
    num_test_examples = len(test_dataset)
    
    print("Data preprocessing complete\n")
    
    return train_loader, test_loader, num_train_examples, num_test_examples



###            DEFINITION OF MODEL SHAPE             ###
"""
Model definition according (approximately) to the YOLO model 
described by Redmon et al. in "You Only Look Once:
Unified, Real-Time Object Detection"
"""
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
        
        # Calculate the size after conv layers for the linear layers
        # For input size 512x512, after all conv and pooling layers, the size will be 4x4
        # 4x4x1024 = 16384
        
        # Define the fully connected layers
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

# Create the model
model = YOLOModel().to(device)

"""
Our final layer predicts bounding box coordinates. We normalize 
the bounding box width and height by the image width and height
so that they fall between 0 and 1.

We use a sigmoid activation function for the final layer to 
facilitate learning of the normalized range of the output.

All the convolution layers use the rectified linear unit activation.
"""

# Custom Loss and metric functions

def IOU_metric(y_true, y_pred, epsilon=1e-10):
    """
    Compute the intersection over the union of the true and
    the predicted bounding boxes. Output in range 0-1;  
    1 being the best match of bounding boxes (perfect alignment), 
    0 being worst (no intersection at all).
    
    @param y_true - BATCH_SIZEx4 Tensor object (float), the ground 
                    truth labels for the bounding boxes around the
                    tumor in the corresponding images. Value order:
                    (x_top_left, y_top_left, x_bot_right, y_bot_right)
    @param y_pred - BATCH_SIZEx4 Tensor object (float), the model's 
                    prediction for the bounding boxes around the
                    tumor in the corresponding images. Value order:
                    (x_top_left, y_top_left, x_bot_right, y_bot_right)
    @param epsilon - small value to avoid division by zero
    @return iou - 1x1 Tensor object (float), value being the mean
                  IOU for all image in the batch, and is within the 
                  range of 0-1 (inclusive). 
    """
    # extract points from tensors
    x_LT = torch.min(y_true[:, 0], y_true[:, 2])
    y_UT = torch.min(y_true[:, 1], y_true[:, 3])
    x_RT = torch.max(y_true[:, 0], y_true[:, 2])
    y_LT = torch.max(y_true[:, 1], y_true[:, 3])

    x_LP = torch.min(y_pred[:, 0], y_pred[:, 2])
    y_UP = torch.min(y_pred[:, 1], y_pred[:, 3])
    x_RP = torch.max(y_pred[:, 0], y_pred[:, 2])
    y_LP = torch.max(y_pred[:, 1], y_pred[:, 3])

    # to perform the IOU math correctly, the points that are left-most,
    # upper-most, right-most, and lower-most must be found
    xL_pairwise_gt = (x_LT > x_LP).float()
    yU_pairwise_gt = (y_UT > y_UP).float()

    xW1_pairwise_int = (x_LT < x_RP).float()
    xW2_pairwise_int = (x_LP < x_RT).float()

    yH1_pairwise_int = (y_UT < y_LP).float()
    yH2_pairwise_int = (y_UP < y_LT).float()

    # find the amount by which the bboxes intersect
    x_does_intersect = xL_pairwise_gt * xW1_pairwise_int + (1.0 - xL_pairwise_gt) * xW2_pairwise_int
    y_does_intersect = yU_pairwise_gt * yH1_pairwise_int + (1.0 - yU_pairwise_gt) * yH2_pairwise_int
    box_does_intersect = x_does_intersect * y_does_intersect

    a = torch.min(x_RP - x_LT, x_RP - x_LP)
    b = torch.min(x_RT - x_LP, x_RT - x_LT)
    c = torch.min(y_LP - y_UT, y_LP - y_UP)
    d = torch.min(y_LT - y_UP, y_LT - y_UT)

    # calculate intersection area
    intersection_width = xL_pairwise_gt * a + (1.0 - xL_pairwise_gt) * b
    intersection_height = yU_pairwise_gt * c + (1.0 - yU_pairwise_gt) * d

    intersection = intersection_width * intersection_height * box_does_intersect
    
    # calculate union area
    area_true = (x_RT - x_LT) * (y_LT - y_UT)
    area_pred = (x_RP - x_LP) * (y_LP - y_UP)
    union = area_true + area_pred - intersection

    # take the mean in order to compress BATCH_SIZEx1 Tensor into a 1x1 Tensor
    iou = torch.mean(intersection / (union + epsilon))  # Add epsilon to avoid division by zero
    return iou

def log_loss(y_true, y_pred):
    """
    An implementation of the Unitbox negative log loss function proposed
    by Yu et al. in "Unitbox: an advanced object detection network".
    This loss function takes advantage of the observation that all the
    points in a bounding box prediction are highly correlated.

    @param y_true - BATCH_SIZEx4 Tensor object (float), the ground 
                    truth labels for the bounding boxes around the
                    tumor in the corresponding images. Value order:
                    (x_top_left, y_top_left, x_bot_right, y_bot_right)
    @param y_pred - BATCH_SIZEx4 Tensor object (float), the model's 
                    prediction for the bounding boxes around the
                    tumor in the corresponding images. Value order:
                    (x_top_left, y_top_left, x_bot_right, y_bot_right)
    @return loss - 1x1 Tensor object (float), valued between 0 and log(epsilon)
    """
    iou = IOU_metric(y_true, y_pred)
    # use epsilon as a replacement for 0 values to prevent NaNs
    # from appearing in the loss computation
    iou = torch.where(iou == 0, torch.tensor(epsilon, device=iou.device), iou)
    # negative log should act as a function that exponentially punishes
    # boxes that have worse IOU (up to value of log(epsilon))
    loss = -torch.log(iou)
    return loss.mean()
    

# custom loss function using aspects of relevant information from the YOLO paper
def YOLO_loss(y_true, y_pred, lambda_coord=5.0):
    """
    An implementation of the Unitbox negative log loss function proposed
    by Yu et al. in "Unitbox: an advanced object detection network".
    This loss function takes advantage of the observation that all the
    points in a bounding box prediction are highly correlated.
    This loss function is very closely related to the mean squared error.

   @param y_true - BATCH_SIZEx4 Tensor object (float), the ground 
                    truth labels for the bounding boxes around the
                    tumor in the corresponding images. Value order:
                    (x_top_left, y_top_left, x_bot_right, y_bot_right)
   @param y_pred - BATCH_SIZEx4 Tensor object (float), the model's 
                    prediction for the bounding boxes around the
                    tumor in the corresponding images. Value order:
                    (x_top_left, y_top_left, x_bot_right, y_bot_right)
   @param lambda_coord - weight for coordinate loss terms
   @return loss - 1x1 Tensor object (float), value range 0-inf
   """
    # extract the vectors of each of the 4 bbox points from each tensor
    x_LT = y_true[:, 0]
    y_UT = y_true[:, 1]
    x_RT = y_true[:, 2]
    y_LT = y_true[:, 3]

    x_LP = y_pred[:, 0]
    y_UP = y_pred[:, 1]
    x_RP = y_pred[:, 2]
    y_LP = y_pred[:, 3]

    # get the square difference between the midpoints of true and pred bboxs
    x_Pmid = x_LP + (x_RP - x_LP) / 2
    x_Tmid = x_LT + (x_RT - x_LT) / 2
    y_Pmid = y_UP + (y_LP - y_UP) / 2
    y_Tmid = y_UT + (y_LT - y_UT) / 2

    x_mid_sqdiff = (x_Pmid - x_Tmid) ** 2
    y_mid_sqdiff = (y_Pmid - y_Tmid) ** 2
    
    first_term = x_mid_sqdiff + y_mid_sqdiff

    # get the square difference between the width and height of true and pred bboxs
    x_Pwidth = torch.sqrt(torch.abs(x_RP - x_LP))
    x_Twidth = torch.sqrt(torch.abs(x_RT - x_LT))
    y_Pheight = torch.sqrt(torch.abs(y_UP - y_LP))
    y_Theight = torch.sqrt(torch.abs(y_UT - y_LT))
    
    second_term = (x_Pwidth - x_Twidth) ** 2 + (y_Pheight - y_Theight) ** 2

    # combine the 2 terms using the predefined lambda value (for the coordinate points)
    # as a weight on the loss
    loss = (first_term + second_term) * lambda_coord
    return loss.mean()






###                 TRAINING FUNCTIONS                 ###

def train_model(model, train_loader, num_train_examples, num_epochs=1, lr=0.00001, 
                lambda_coord=5.0, epsilon=1e-10, device='cpu', verbose=True):
    """
    Train the YOLO model.
    
    @param model - PyTorch model to train
    @param train_loader - training data loader
    @param num_train_examples - number of training examples
    @param num_epochs - number of epochs to train
    @param lr - learning rate
    @param lambda_coord - YOLO loss coordinator weight
    @param epsilon - epsilon for numerical stability
    @param device - device to train on ('cpu' or 'cuda')
    @param verbose - whether to print training progress
    @return - training history (loss, iou, mse per epoch)
    """
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    print('Fitting the model\n')
    
    training_history = []
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_iou = 0.0
        running_mse = 0.0
        
        from tqdm import tqdm
        epoch_desc = f"Epoch {epoch+1}/{num_epochs}"
        if verbose:
            epoch_desc += " - Training"
        
        for images, targets in tqdm(train_loader, desc=epoch_desc):
            # Move data to device
            images = images.to(device)
            targets = targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = YOLO_loss(targets, outputs, lambda_coord)
            iou = IOU_metric(targets, outputs, epsilon)
            mse = nn.functional.mse_loss(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update running metrics
            running_loss += loss.item() * images.size(0)
            running_iou += iou.item() * images.size(0)
            running_mse += mse.item() * images.size(0)
        
        # Calculate epoch metrics
        epoch_loss = running_loss / num_train_examples
        epoch_iou = running_iou / num_train_examples
        epoch_mse = running_mse / num_train_examples
        
        if verbose:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, IOU: {epoch_iou:.4f}, MSE: {epoch_mse:.4f}")
        
        training_history.append({
            'epoch': epoch + 1,
            'loss': epoch_loss,
            'iou': epoch_iou,
            'mse': epoch_mse
        })
    
    return training_history



###                 EVALUATION FUNCTIONS               ###

def evaluate_model(model, test_loader, num_test_examples, lambda_coord=5.0, 
                   epsilon=1e-10, device='cpu', verbose=True):
    """
    Evaluate the YOLO model on test data.
    
    @param model - PyTorch model to evaluate
    @param test_loader - test data loader
    @param num_test_examples - number of test examples
    @param lambda_coord - YOLO loss coordinator weight
    @param epsilon - epsilon for numerical stability
    @param device - device to evaluate on ('cpu' or 'cuda')
    @param verbose - whether to print evaluation progress
    @return - evaluation metrics (loss, iou, mse)
    """
    print('Evaluating the model\n')
    
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        test_loss = 0.0
        test_iou = 0.0
        test_mse = 0.0
        
        from tqdm import tqdm
        eval_desc = "Evaluating"
        if verbose:
            eval_desc += " model"
        
        for images, targets in tqdm(test_loader, desc=eval_desc):
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            
            loss = YOLO_loss(targets, outputs, lambda_coord)
            iou = IOU_metric(targets, outputs, epsilon)
            mse = nn.functional.mse_loss(outputs, targets)
            
            test_loss += loss.item() * images.size(0)
            test_iou += iou.item() * images.size(0)
            test_mse += mse.item() * images.size(0)
        
        # Calculate test metrics
        test_loss /= num_test_examples
        test_iou /= num_test_examples
        test_mse /= num_test_examples
        
        if verbose:
            print(f"Test Loss: {test_loss:.4f}, Test IOU: {test_iou:.4f}, Test MSE: {test_mse:.4f}")
        
        return {
            'loss': test_loss,
            'iou': test_iou,
            'mse': test_mse
        }



###                 MODEL SAVING FUNCTIONS             ###

def save_model(model, weight_path):
    """
    Save the trained model weights.
    
    @param model - PyTorch model to save
    @param weight_path - path to save the model weights
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(weight_path), exist_ok=True)
    
    # Save only the model weights
    torch.save(model.state_dict(), weight_path)
    print(f"Saved model weights to {weight_path}")

###                 MAIN FUNCTION                     ###

def main():
    """
    Main function to run the training and evaluation pipeline.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Load and prepare data
    train_loader, test_loader, num_train_examples, num_test_examples = load_and_prepare_data(
        csv_path=args.csv_path,
        image_base_path=args.image_path,
        test_size=args.test_size,
        mirror=args.mirror,
        batch_size=args.batch_size
    )
    
    # Create the model
    model = YOLOModel().to(device)
    
    if args.verbose:
        print(f"Model created on device: {device}")
        print(f"Training examples: {num_train_examples}")
        print(f"Test examples: {num_test_examples}")
    
    # Train the model
    training_history = train_model(
        model=model,
        train_loader=train_loader,
        num_train_examples=num_train_examples,
        num_epochs=args.epochs,
        lr=args.lr,
        lambda_coord=args.lambda_coord,
        epsilon=args.epsilon,
        device=device,
        verbose=args.verbose
    )
    
    # Evaluate the model
    eval_metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        num_test_examples=num_test_examples,
        lambda_coord=args.lambda_coord,
        epsilon=args.epsilon,
        device=device,
        verbose=args.verbose
    )
    
    # Save the model if requested
    if args.save_model:
        save_model(model, args.weight_path)
    
    if args.verbose:
        print("\nTraining completed successfully!")
        print(f"Final training loss: {training_history[-1]['loss']:.4f}")
        print(f"Final training IOU: {training_history[-1]['iou']:.4f}")
        print(f"Test loss: {eval_metrics['loss']:.4f}")
        print(f"Test IOU: {eval_metrics['iou']:.4f}")

if __name__ == "__main__":
    main()

