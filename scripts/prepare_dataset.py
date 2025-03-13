# File: scripts/prepare_dataset.py
# Script chuẩn bị dữ liệu

import os
import argparse
import shutil
import random
from tqdm import tqdm
from src.data.preprocessing import preprocess_dataset
from src.data.augmentation import augment_dataset

def setup_directory_structure(base_dir):
    """Thiết lập cấu trúc thư mục dự án."""
    # Tạo thư mục chính
    os.makedirs(base_dir, exist_ok=True)
    
    # Tạo các thư mục con
    for directory in ['data/raw', 'data/processed', 'data/augmented', 'models', 'logs', 'evaluation_results']:
        os.makedirs(os.path.join(base_dir, directory), exist_ok=True)

def organize_raw_data(input_dir, output_dir, disease_classes):
    """Sắp xếp dữ liệu thô theo cấu trúc chuẩn."""
    # Tạo thư mục đầu ra
    os.makedirs(output_dir, exist_ok=True)
    
    # Tạo thư mục cho từng lớp bệnh
    for disease in disease_classes:
        os.makedirs(os.path.join(output_dir, disease), exist_ok=True)
    
    # Kiểm tra nếu input_dir là thư mục hoặc file nén
    if os.path.isdir(input_dir):
        # Trường hợp 1: input_dir đã có cấu trúc thư mục theo bệnh
        if all(os.path.isdir(os.path.join(input_dir, d)) for d in disease_classes):
            print("Input directory already has the correct structure. Copying files...")
            
            # Copy file từ thư mục input sang thư mục output
            for disease in disease_classes:
                src_dir = os.path.join(input_dir, disease)
                dst_dir = os.path.join(output_dir, disease)
                
                # Lấy danh sách file ảnh
                image_files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                # Copy từng file
                for file in tqdm(image_files, desc=f"Copying {disease}"):
                    shutil.copy(os.path.join(src_dir, file), os.path.join(dst_dir, file))
        
        # Trường hợp 2: input_dir chứa tất cả ảnh và tên file chứa thông tin về bệnh
        else:
            print("Organizing files based on filename...")
            
            # Lấy danh sách file ảnh
            image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Phân loại file dựa trên tên
            for file in tqdm(image_files, desc="Organizing files"):
                # Kiểm tra tên file chứa tên bệnh
                for disease in disease_classes:
                    if disease.lower() in file.lower():
                        # Copy file vào thư mục tương ứng
                        shutil.copy(
                            os.path.join(input_dir, file),
                            os.path.join(output_dir, disease, file)
                        )
                        break

def main():
    parser = argparse.ArgumentParser(description='Prepare mango disease dataset')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing raw data')
    parser.add_argument('--output_dir', type=str, default='data', help='Output directory for processed data')
    parser.add_argument('--project_dir', type=str, default='.', help='Project root directory')
    parser.add_argument('--disease_classes', type=str, nargs='+', 
                        default=['anthracnose', 'bacterial_canker', 'healthy', 'powdery_mildew'],
                        help='List of disease classes')
    parser.add_argument('--img_size', type=int, nargs=2, default=[224, 224], help='Image size (width, height)')
    parser.add_argument('--test_split', type=float, default=0.15, help='Proportion of test data')
    parser.add_argument('--validation_split', type=float, default=0.15, help='Proportion of validation data')
    parser.add_argument('--augment', action='store_true', help='Apply data augmentation')
    
    args = parser.parse_args()
    
    # Thiết lập cấu trúc thư mục dự án
    setup_directory_structure(args.project_dir)
    
    # Sắp xếp dữ liệu thô
    raw_data_dir = os.path.join(args.output_dir, 'raw')
    organize_raw_data(args.input_dir, raw_data_dir, args.disease_classes)
    
    # Tiền xử lý dữ liệu
    processed_data_dir = os.path.join(args.output_dir, 'processed')
    preprocess_dataset(
        input_dir=raw_data_dir,
        output_dir=processed_data_dir,
        img_size=tuple(args.img_size),
        test_split=args.test_split,
        validation_split=args.validation_split
    )
    
    # Tăng cường dữ liệu nếu được yêu cầu
    if args.augment:
        augmented_data_dir = os.path.join(args.output_dir, 'augmented')
        
        # Cấu hình tăng cường dữ liệu
        augmentation_config = {
            'horizontal_flip': True,
            'vertical_flip': False,
            'rotation': True,
            'rotation_limit': 20,
            'random_brightness_contrast': True,
            'brightness_limit': 0.2,
            'contrast_limit': 0.2,
            'gaussian_blur': True,
            'blur_limit': 7,
            'gaussian_noise': True,
            'noise_var_limit': (10.0, 50.0)
        }
        
        # Thực hiện tăng cường dữ liệu
        augment_dataset(
            input_dir=os.path.join(processed_data_dir, 'train'),
            output_dir=augmented_data_dir,
            augmentation_config=augmentation_config,
            num_augmentations=5
        )
    
    print("Dataset preparation completed!")

if __name__ == "__main__":
    main()