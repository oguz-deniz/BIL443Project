import os
import cv2

def apply_clahe(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray_image)
    
    # Save the processed image to the output path
    cv2.imwrite(output_path, clahe_image)

def preprocess_dataset(root_dir, output_dir):
    # Create output directories if they do not exist
    train_output_dir = os.path.join(output_dir, 'train', 'images')
    val_output_dir = os.path.join(output_dir, 'val', 'images')
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(val_output_dir, exist_ok=True)
    
    # Process train images
    train_dir = os.path.join(root_dir, 'train', 'images')
    for filename in os.listdir(train_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(train_dir, filename)
            output_path = os.path.join(train_output_dir, filename)
            apply_clahe(image_path, output_path)
    
    # Process val images
    val_dir = os.path.join(root_dir, 'val', 'images')
    for filename in os.listdir(val_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(val_dir, filename)
            output_path = os.path.join(val_output_dir, filename)
            apply_clahe(image_path, output_path)

if __name__ == '__main__':
    root_dir = 'dataset_yolo'  
    output_dir = 'clahed_dataset_yolo'  
    preprocess_dataset(root_dir, output_dir)
