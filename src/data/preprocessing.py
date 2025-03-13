# File: src/data/preprocessing.py
# Tiền xử lý dữ liệu

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm

def create_directory_structure(base_dir, class_names):
    """Tạo cấu trúc thư mục cho dataset đã xử lý."""
    # Tạo thư mục gốc nếu chưa tồn tại
    os.makedirs(base_dir, exist_ok=True)
    
    # Tạo thư mục train, validation, test
    for split in ['train', 'validation', 'test']:
        split_dir = os.path.join(base_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        # Tạo thư mục cho từng loại bệnh
        for class_name in class_names:
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

def preprocess_image(image, img_size):
    """Tiền xử lý một ảnh đơn lẻ."""
    # Resize ảnh
    image = cv2.resize(image, img_size)
    
    # Chuyển đổi BGR sang RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Chuẩn hóa giá trị pixel về [0, 1]
    image = image / 255.0
    
    return image

def preprocess_dataset(input_dir, output_dir, img_size=(224, 224), test_split=0.15, validation_split=0.15):
    """Tiền xử lý toàn bộ dataset."""
    # Lấy danh sách các lớp (thư mục con của input_dir)
    class_names = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    # Tạo cấu trúc thư mục đầu ra
    create_directory_structure(output_dir, class_names)
    
    # Xử lý từng lớp
    for class_name in class_names:
        class_dir = os.path.join(input_dir, class_name)
        
        # Lấy tất cả các file ảnh trong thư mục
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Chia tập dữ liệu
        train_files, test_files = train_test_split(image_files, test_size=test_split, random_state=42)
        train_files, val_files = train_test_split(train_files, test_size=validation_split/(1-test_split), random_state=42)
        
        # Xử lý và lưu ảnh cho tập train
        for split, files in [('train', train_files), ('validation', val_files), ('test', test_files)]:
            for file in tqdm(files, desc=f"Processing {class_name} - {split}"):
                # Đọc ảnh
                img_path = os.path.join(class_dir, file)
                image = cv2.imread(img_path)
                
                if image is None:
                    print(f"Warning: Cannot read {img_path}")
                    continue
                
                # Tiền xử lý ảnh
                processed_image = preprocess_image(image, img_size)
                
                # Lưu ảnh đã xử lý
                output_path = os.path.join(output_dir, split, class_name, file)
                cv2.imwrite(output_path, cv2.cvtColor(processed_image * 255, cv2.COLOR_RGB2BGR))
    
    print(f"Preprocessing complete. Data saved to {output_dir}")
    
def process_json_to_mask(json_path, output_size=(512, 512), label_mapping=None):
    """Chuyển đổi file JSON annotation thành mask."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Lấy kích thước ảnh gốc
        img_height = data.get('imageHeight', output_size[0])
        img_width = data.get('imageWidth', output_size[1])
        
        print(f"Processing JSON: {json_path}")
        print(f"Original size: {img_width}x{img_height}")
        
        # Tạo mask trống
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        
        # Vẽ các đa giác lên mask
        for shape in data.get('shapes', []):
            label = shape.get('label')
            points = shape.get('points')
            
            print(f"  Shape label: {label}, points: {len(points)}")
            
            # Chuyển đổi label thành ID nếu có mapping
            label_id = label_mapping.get(label, 1) if label_mapping else 1
            print(f"  Mapped to ID: {label_id}")
            
            # Kiểm tra points
            if len(points) < 3:
                print(f"  Warning: Not enough points ({len(points)}) for polygon")
                continue
                
            # Chuyển đổi points thành định dạng phù hợp cho cv2.fillPoly
            points_array = np.array(points, dtype=np.int32)
            
            # Vẽ polygon
            cv2.fillPoly(mask, [points_array], label_id)
            
            # Kiểm tra xem polygon có được vẽ không
            pixel_count = np.sum(mask == label_id)
            print(f"  Filled pixels: {pixel_count}")
            
            if pixel_count == 0:
                print(f"  Warning: No pixels set for label {label}")
                print(f"  Points sample: {points[:3]}")
        
        # Lưu mask gốc để kiểm tra (chỉ khi debug)
        debug_dir = "debug_masks"
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, os.path.basename(json_path) + ".png")
        cv2.imwrite(debug_path, mask * 50)  # Nhân với 50 để dễ nhìn thấy
        print(f"  Saved debug mask to: {debug_path}")
        
        # Resize mask về kích thước mong muốn
        if output_size and (img_height != output_size[0] or img_width != output_size[1]):
            mask = cv2.resize(mask, output_size, interpolation=cv2.INTER_NEAREST)
            
            # Lưu mask sau resize để kiểm tra
            resized_debug_path = os.path.join(debug_dir, f"resized_{os.path.basename(json_path)}.png")
            cv2.imwrite(resized_debug_path, mask * 50)
            print(f"  Saved resized mask to: {resized_debug_path}")
        
        return mask
    
    except Exception as e:
        print(f"Lỗi khi xử lý file {json_path}: {e}")
        import traceback
        traceback.print_exc()
        return None