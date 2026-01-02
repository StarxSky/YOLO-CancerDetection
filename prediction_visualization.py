# Import necessary libraries
import os
import glob
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import traceback

class ImageViewer:
    def __init__(self, root, image_dir, image_formats=('*.png', '*.jpg', '*.jpeg', '*.bmp')):
        self.root = root
        self.root.title("Prediction Images Viewer")
        self.root.geometry("800x600")
        
        # Set the directory containing images
        self.image_dir = image_dir
        
        # Get list of all image files in the directory
        self.image_files = []
        for fmt in image_formats:
            self.image_files.extend(glob.glob(os.path.join(image_dir, fmt)))
        
        # Sort files alphabetically/numerically
        self.image_files.sort()
        
        # Current image index
        self.current_index = 0
        
        # Create UI components
        self.create_widgets()
        
        # Load the first image if available
        if self.image_files:
            self.load_image(self.current_index)
        else:
            self.status_label.config(text="No images found in the directory")
    
    def create_widgets(self):
        # Create a frame for the image
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a canvas to display the image
        self.canvas = tk.Canvas(self.image_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Create a frame for navigation buttons
        self.nav_frame = tk.Frame(self.root)
        self.nav_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Previous button
        self.prev_button = ttk.Button(self.nav_frame, text="Previous", command=self.show_previous)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        # Next button
        self.next_button = ttk.Button(self.nav_frame, text="Next", command=self.show_next)
        self.next_button.pack(side=tk.RIGHT, padx=5)
        
        # Status label
        self.status_label = ttk.Label(self.root, text="", relief=tk.SUNKEN, anchor=tk.CENTER)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
    
    def load_image(self, index):
        if 0 <= index < len(self.image_files):
            # Get the image file path
            image_path = self.image_files[index]
            
            # Open the image
            self.current_image = Image.open(image_path)
            
            # Resize image to fit the canvas while maintaining aspect ratio
            self.resize_image()
            
            # Display the image
            self.tk_image = ImageTk.PhotoImage(self.resized_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
            
            # Update status label
            self.status_label.config(text=f"Image {index+1}/{len(self.image_files)}: {os.path.basename(image_path)}")
    
    def resize_image(self):
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # If canvas not yet rendered, use default size
        if canvas_width == 1 or canvas_height == 1:
            canvas_width = 780
            canvas_height = 520
        
        # Calculate the aspect ratio
        img_width, img_height = self.current_image.size
        aspect_ratio = img_width / img_height
        
        # Calculate new size that fits in the canvas
        if aspect_ratio > 1:
            # Image is wider than tall
            new_width = min(canvas_width, img_width)
            new_height = int(new_width / aspect_ratio)
        else:
            # Image is taller than wide or square
            new_height = min(canvas_height, img_height)
            new_width = int(new_height * aspect_ratio)
        
        # Resize the image
        self.resized_image = self.current_image.resize((new_width, new_height), Image.LANCZOS)
    
    def show_previous(self):
        if self.image_files:
            self.current_index = (self.current_index - 1) % len(self.image_files)
            self.load_image(self.current_index)
    
    def show_next(self):
        if self.image_files:
            self.current_index = (self.current_index + 1) % len(self.image_files)
            self.load_image(self.current_index)
    
    def on_resize(self, event):
        # Resize image when window size changes
        if self.image_files:
            self.resize_image()
            self.tk_image = ImageTk.PhotoImage(self.resized_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

def show_predictions_with_matplotlib(image_dir):
    """
    Display prediction images using matplotlib, suitable for headless environments like Jupyter notebooks.
    """
    # Get list of all image files in the directory
    image_formats = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
    image_files = []
    for fmt in image_formats:
        image_files.extend(glob.glob(os.path.join(image_dir, fmt)))
    
    # Sort files alphabetically/numerically
    image_files.sort()
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    # Calculate grid size for displaying images
    num_images = len(image_files)
    num_cols = 2  # Adjust as needed
    num_rows = (num_images + num_cols - 1) // num_cols  # Ceiling division
    
    # Create figure and subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 8 * num_rows))
    
    # Flatten axes for easy iteration
    axes = axes.flatten() if num_rows > 1 or num_cols > 1 else [axes]
    
    # Display each image
    for i, (ax, img_path) in enumerate(zip(axes, image_files)):
        # Read the image
        img = mpimg.imread(img_path)
        
        # Display the image
        ax.imshow(img)
        
        # Set title with image filename
        ax.set_title(f"{os.path.basename(img_path)}")
        
        # Hide axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
    # Hide any unused subplots
    for i in range(len(image_files), len(axes)):
        axes[i].axis('off')
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()


def main():
    # Directory containing prediction images
    image_dir = "predictions"
    
    # Check if the directory exists
    if not os.path.exists(image_dir):
        print(f"Error: Directory '{image_dir}' not found.")
        print("Please run the inference.py script first to generate prediction images.")
        return
    
    # Check if Tkinter is available and can be initialized
    tk_available = True
    root = None
    
    try:
        # Try to create a Tk root window
        root = tk.Tk()
        # Hide the window immediately
        root.withdraw()
        # Destroy the window
        root.destroy()
    except (tk.TclError, AttributeError) as e:
        print(f"Tkinter is not available: {e}")
        tk_available = False
    
    # Use Tkinter if available, otherwise fall back to matplotlib
    if tk_available:
        try:
            # Create the main window
            root = tk.Tk()
            
            # Create the image viewer
            viewer = ImageViewer(root, image_dir)
            
            # Bind resize event
            root.bind("<Configure>", viewer.on_resize)
            
            # Run the application
            root.mainloop()
        except (tk.TclError, AttributeError) as e:
            print(f"Tkinter failed to initialize: {e}")
            print("Falling back to matplotlib for image display.")
            show_predictions_with_matplotlib(image_dir)
    else:
        print("Using matplotlib for image display (Tkinter not available).")
        show_predictions_with_matplotlib(image_dir)

if __name__ == "__main__":
    main()
