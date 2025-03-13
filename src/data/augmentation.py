# File: src/data/augmentation.py
# Tăng cường dữ liệu

import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

def get_augmentation_pipeline(config):
    """Tạo pipeline tăng cường dữ liệu dựa trên cấu hình."""
    aug_list = []
    
    if config.get('horizontal_flip', False):
        aug_list.append(A.HorizontalFlip(p=0.5))
    
    if config.get('vertical_flip', False):
        aug_list.append(A.VerticalFlip(p=0.5))
    
    if config.get('rotation', False):
        aug_list.append(A.Rotate(limit=config.get('rotation_limit', 20), p=0.7))
    
    if config.get('random_brightness_contrast', False):
        aug_list.append(A.RandomBrightnessContrast(
            brightness_limit=config.get('brightness_limit', 0.2),
            contrast_limit=config.get('contrast_limit', 0.2),
            p=0.7
        ))
    
    if config.get('gaussian_blur', False):
        aug_list.append(A.GaussianBlur(
            blur_limit=config.get('blur_limit', 7),
            p=0.3
        ))
    
    if config.get('gaussian_noise', False):
        aug_list.append(A.GaussianNoise(
            var_limit=config.get('noise_var_limit', (10.0, 50.0)),
            p=0.3
        ))
    
    return A.Compose(aug_list)

def augment_dataset(input_dir, output_dir, augmentation_config, num_augmentations=5):
    """Tăng cường dataset."""
    # Tạo thư mục đầu ra
    os.makedirs(output_dir, exist_ok=True)
    
    # Lấy danh sách các lớp
    class_names = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    # Tạo pipeline tăng cường
    aug_pipeline = get_augmentation_pipeline(augmentation_config)
    
    # Xử lý từng lớp
    for class_name in class_names:
        class_dir = os.path.join(input_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        
        # Tạo thư mục đầu ra cho lớp
        os.makedirs(output_class_dir, exist_ok=True)
        
        # Lấy tất cả các file ảnh trong thư mục
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Xử lý từng ảnh
        for file in tqdm(image_files, desc=f"Augmenting {class_name}"):
            # Đọc ảnh
            img_path = os.path.join(class_dir, file)
            image = cv2.imread(img_path)
            
            if image is None:
                print(f"Warning: Cannot read {img_path}")
                continue
            
            # Chuyển sang RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Copy ảnh gốc sang thư mục đầu ra
            output_path = os.path.join(output_class_dir, file)
            cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            # Tạo nhiều phiên bản tăng cường
            for i in range(num_augmentations):
                # Áp dụng tăng cường
                augmented = aug_pipeline(image=image)['image']
                
                # Lưu ảnh đã tăng cường
                file_name, file_ext = os.path.splitext(file)
                output_path = os.path.join(output_class_dir, f"{file_name}_aug_{i}{file_ext}")
                cv2.imwrite(output_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
    
    print(f"Augmentation complete. Data saved to {output_dir}")