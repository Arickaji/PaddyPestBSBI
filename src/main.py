import json
import os
from PIL import Image
import numpy as np
import albumentations as A
import cv2
import matplotlib.pyplot as plt

def load_coco_annotations(annotation_file):
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    return data

def create_mask_from_annotations(image_id, annotations, image_shape):
    """Create a binary mask from COCO annotations for a specific image"""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    # Find all annotations for this image
    image_annotations = [ann for ann in annotations if ann['image_id'] == image_id]
    
    # Draw each segmentation onto the mask
    for ann in image_annotations:
        # Get segmentation polygons
        if 'segmentation' in ann:
            for segmentation in ann['segmentation']:
                # Convert to numpy array and reshape
                pts = np.array(segmentation).reshape((-1, 2)).astype(np.int32)
                # Draw filled polygon
                cv2.fillPoly(mask, [pts], 1)
    
    return mask

def augment_image_and_mask(image, mask):
    # Ensure inputs are numpy arrays
    if image is None or mask is None:
        raise ValueError("Image or mask is None")
    
    # Make sure mask is 2D (height, width)
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
        
    augmented = augmentation_pipeline(image=image, mask=mask)
    return augmented['image'], augmented['mask']

def process_all_images(coco_data, images_dir):
    """Process and display all images with their IDs and masks"""
    # Get all image IDs and filenames
    image_files = {img['id']: img['file_name'] for img in coco_data['images']}
    
    # Calculate grid dimensions
    n_images = len(image_files)
    n_cols = 4  # You can adjust this number
    n_rows = (n_images + n_cols - 1) // n_cols
    
    # Create figure
    plt.figure(figsize=(20, 5 * n_rows))
    
    for idx, (image_id, filename) in enumerate(sorted(image_files.items())):
        try:
            # Load image
            image_path = os.path.join(images_dir, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load image {filename}")
                continue
                
            # Create mask
            mask = create_mask_from_annotations(image_id, coco_data['annotations'], image.shape)
            
            # Apply augmentation
            aug_image, aug_mask = augment_image_and_mask(image, mask)
            
            # Plot original and augmented images
            # Original image
            plt.subplot(n_rows, n_cols * 2, idx * 2 + 1)
            plt.title(f'ID: {image_id} - Original')
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
            # Augmented image with mask overlay
            plt.subplot(n_rows, n_cols * 2, idx * 2 + 2)
            plt.title(f'ID: {image_id} - Augmented')
            
            # Create a mask overlay on the augmented image
            aug_image_with_mask = aug_image.copy()
            aug_image_with_mask[aug_mask > 0] = [255, 0, 0]  # Red overlay for mask
            
            plt.imshow(cv2.cvtColor(aug_image_with_mask, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            
        except Exception as e:
            print(f"Error processing image {filename}: {str(e)}")
            continue
    
    plt.tight_layout()
    return plt.gcf()  # Return the figure

# Directory paths
dataset_dir = "../data"  # Change this to point to your data directory
annotations_path = os.path.join(dataset_dir, "PaddyPestData.json")
images_dir = os.path.join(dataset_dir, "images")

# Add check for file existence
if not os.path.exists(annotations_path):
    raise FileNotFoundError(f"Annotation file not found at {annotations_path}. Please ensure PaddyPestData.json is in the {dataset_dir} directory.")

if not os.path.exists(images_dir):
    raise FileNotFoundError(f"Images directory not found at {images_dir}. Please ensure your images are in the {dataset_dir}/images directory.")

# Load COCO annotations
coco_data = load_coco_annotations(annotations_path)
print(f"Loaded {len(coco_data['images'])} images.")

# Define the augmentation pipeline
augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
    A.RandomBrightnessContrast(p=0.5)
], additional_targets={'mask': 'mask'})

try:
    # Process and display all images
    fig = process_all_images(coco_data, images_dir)
    
    # Save the figure
    output_path = 'augmentation_report.png'
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Report saved as {output_path}")
    
    # Display the figure
    plt.show()

except Exception as e:
    print(f"An error occurred: {str(e)}")

