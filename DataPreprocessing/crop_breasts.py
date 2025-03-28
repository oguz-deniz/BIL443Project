import os
import cv2
import numpy as np

def crop_image(image):
   
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) > 127:
        # White background
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    else:
        # Black background
        _, binary = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image, x, y, w, h

def find_mask(mask_dir, category, patient_id, view):
    for subdir in ['Normal_masks', 'Kitle_masks', 'Kalsifikasyon_masks']:
        mask_filename = f"{patient_id}_{view}_mask.png"
        mask_path = os.path.join(mask_dir, category, subdir, mask_filename)
        if os.path.exists(mask_path):
            return mask_path, subdir
    return None, None

def process_directory(image_dir, mask_dir, output_image_dir, output_mask_dir):
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith('.png'):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                cropped_image, x, y, w, h = crop_image(image)
                
                relative_path = os.path.relpath(image_path, image_dir)
                output_image_path = os.path.join(output_image_dir, relative_path)
                os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
                cv2.imwrite(output_image_path, cropped_image)
                
                category = os.path.basename(os.path.dirname(root)).replace("_png","_masks")
                
                view = os.path.splitext(file)[0].split('_')[0]
                patient_id = os.path.basename(os.path.dirname(relative_path))
                mask_path, subdir = find_mask(mask_dir, category, patient_id, view)
 
                
                if mask_path:
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    cropped_mask = mask[y:y+h, x:x+w]
                    output_mask_path = os.path.join(output_mask_dir, category, subdir, os.path.basename(mask_path))
                    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
                    cv2.imwrite(output_mask_path, cropped_mask)


image_dir = r'Images_png'
mask_dir = r'Images_masks_unseperated'
output_image_dir = r'Images_cropped_png'
output_mask_dir = r'Images_cropped_masks_png'

process_directory(image_dir, mask_dir, output_image_dir, output_mask_dir)
