import albumentations as A
import cv2
import os
import numpy as np

# Define the augmentation pipeline
augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRoxtate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
    A.RandomBrightnessContrast(p=0.5)
], additional_targets={'mask': 'mask'})

def augment_image_and_mask(image, mask):
    # Ensure inputs are numpy arrays
    if image is None or mask is None:
        raise ValueError("Image or mask is None. Check file paths.")
    
    # Ensure mask is a numpy array
    if not isinstance(mask, np.ndarray):
        raise TypeError("Mask must be a numpy array")
    
    # Make sure mask is 2D (height, width)
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]  # Take first channel if mask is 3D
        
    augmented = augmentation_pipeline(image=image, mask=mask)
    return augmented['image'], augmented['mask']

# Load and check images
def load_image_and_mask(image_path, mask_path):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")
    if mask is None:
        raise FileNotFoundError(f"Could not load mask from {mask_path}")
        
    return image, mask

# Example usage
try:
    # Make sure these paths exist and are correct
    images_dir = 'path/to/your/images'  # Replace with your actual images directory
    masks_dir = 'path/to/your/masks'    # Replace with your actual masks directory
    
    image_path = os.path.join(images_dir, 'image_1.jpg')  # Replace with your actual image filename
    mask_path = os.path.join(masks_dir, 'mask_1.png')     # Replace with your actual mask filename
    
    # Load images
    image, mask = load_image_and_mask(image_path, mask_path)
    
    # Apply augmentation
    aug_image, aug_mask = augment_image_and_mask(image, mask)
    
    # Visualize augmentation (requires matplotlib)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.title('Augmented Image')
    plt.imshow(cv2.cvtColor(aug_image, cv2.COLOR_BGR2RGB))
    plt.subplot(1,2,2)
    plt.title('Augmented Mask')
    plt.imshow(aug_mask, cmap='gray')
    plt.show()

except Exception as e:
    print(f"An error occurred: {str(e)}")
