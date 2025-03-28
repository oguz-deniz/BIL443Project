import pydicom
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def normalize_image(img_array):
    img_array = img_array.astype(np.float32)
    img_array -= img_array.min()
    img_array /= img_array.max()
    img_array *= 255
    return img_array.astype(np.uint8)

def convert_dcm_to_png(dicom_folder, output_base_folder):
    for category in os.listdir(dicom_folder):
        category_path = os.path.join(dicom_folder, category)
        if os.path.isdir(category_path):
            output_category_folder = os.path.join(output_base_folder, f"{category}_png")
            if not os.path.exists(output_category_folder):
                os.makedirs(output_category_folder)
            
            for root, dirs, files in os.walk(category_path):
                for file in files:
                    if file.endswith('.dcm'):
                        dicom_path = os.path.join(root, file)
                        ds = pydicom.dcmread(dicom_path)
                        img_array = ds.pixel_array
                       
                        img_array_normalized = normalize_image(img_array)
                        img = Image.fromarray(img_array_normalized)

                        if img.mode != 'L':
                            img = img.convert('L')
                        
                        relative_path = os.path.relpath(root, category_path)
                        output_patient_folder = os.path.join(output_category_folder, relative_path)
                        
                        if not os.path.exists(output_patient_folder):
                            os.makedirs(output_patient_folder)
                        
                        output_filename = file.replace('.dcm', '.png')
                        output_path = os.path.join(output_patient_folder, output_filename)
                        img.save(output_path)
                        
                        #display_image(img_array_normalized, output_filename)
                        

def display_image(img_array, title):
    plt.imshow(img_array, cmap='gray')
    plt.title(f'Image: {title}')
    plt.axis('off')   
    plt.show()


dicom_folder = 'Images'
output_base_folder = 'Images-converted_png'
convert_dcm_to_png(dicom_folder, output_base_folder)
