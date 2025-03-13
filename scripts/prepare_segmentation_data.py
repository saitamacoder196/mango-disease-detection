# File: scripts/prepare_segmentation_data.py
# Script chuẩn bị dữ liệu phân đoạn cho xoài

import os
import json
import shutil
import argparse
from tqdm import tqdm
from pathlib import Path
import random
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

def setup_directory_structure(base_dir):
    """Thiết lập cấu trúc thư mục cho dữ liệu phân đoạn."""
    # Tạo thư mục chính
    os.makedirs(base_dir, exist_ok=True)
    
    # Tạo các thư mục con
    for directory in ['raw', 'segmentation/images', 'segmentation/masks', 'segmentation/annotations',
                      'segmentation/train/images', 'segmentation/train/masks',
                      'segmentation/val/images', 'segmentation/val/masks',
                      'segmentation/test/images', 'segmentation/test/masks']:
        os.makedirs(os.path.join(base_dir, directory), exist_ok=True)
    
    print(f"Cấu trúc thư mục đã được tạo tại {base_dir}")

def collect_data_files(input_dir):
    """Thu thập tất cả các cặp file ảnh và json từ thư mục đầu vào."""
    print(f"Đang quét thư mục {input_dir} để tìm file ảnh và annotation...")
    
    image_files = []
    json_files = []
    
    # Duyệt qua tất cả các thư mục và tìm file
    for root, _, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            # Thu thập file ảnh
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(file_path)
            # Thu thập file json
            elif file.lower().endswith('.json'):
                json_files.append(file_path)
    
    print(f"Đã tìm thấy {len(image_files)} file ảnh và {len(json_files)} file annotation.")
    return image_files, json_files

def match_image_annotation(image_files, json_files):
    """Ghép cặp file ảnh và file annotation."""
    print("Đang ghép cặp file ảnh và annotation...")
    
    # Tạo dict lưu tên file ảnh và đường dẫn
    image_dict = {}
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        image_dict[img_name] = img_path
    
    # Tìm file json tương ứng với mỗi ảnh
    matched_pairs = []
    for json_path in json_files:
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                json_data = json.load(f)
                # Lấy tên file ảnh từ imagePath trong json
                image_path = json_data.get('imagePath', '')
                if not image_path:
                    continue
                
                # Chuẩn hóa đường dẫn và lấy tên file
                image_name = os.path.basename(image_path.replace('\\', '/'))
                
                # Tìm file ảnh tương ứng
                if image_name in image_dict:
                    matched_pairs.append((image_dict[image_name], json_path))
                else:
                    # Trường hợp tên file trong json không khớp chính xác
                    # Tìm file có tên gần giống
                    potential_matches = [img for img in image_dict.keys() 
                                        if os.path.splitext(img)[0] in os.path.splitext(image_name)[0] 
                                        or os.path.splitext(image_name)[0] in os.path.splitext(img)[0]]
                    if potential_matches:
                        matched_pairs.append((image_dict[potential_matches[0]], json_path))
            except json.JSONDecodeError:
                print(f"Lỗi khi đọc file {json_path}. Bỏ qua.")
    
    print(f"Đã ghép được {len(matched_pairs)} cặp ảnh và annotation.")
    return matched_pairs

def process_json_to_mask(json_path, output_size=(512, 512), label_mapping=None):
    """Chuyển đổi file JSON annotation thành mask."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Lấy kích thước ảnh gốc
        img_height = data.get('imageHeight', output_size[0])
        img_width = data.get('imageWidth', output_size[1])
        
        # Tạo mask trống
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        
        # Vẽ các đa giác lên mask
        for shape in data.get('shapes', []):
            label = shape.get('label')
            points = shape.get('points')
            
            # Chuyển đổi label thành ID nếu có mapping
            label_id = label_mapping.get(label, 1) if label_mapping else 1
            
            # Chuyển đổi points thành định dạng phù hợp cho cv2.fillPoly
            points_array = np.array(points, dtype=np.int32)
            
            # Vẽ polygon
            cv2.fillPoly(mask, [points_array], label_id)
        
        # Resize mask về kích thước mong muốn
        if output_size and (img_height != output_size[0] or img_width != output_size[1]):
            mask = cv2.resize(mask, output_size, interpolation=cv2.INTER_NEAREST)
        
        return mask
    
    except Exception as e:
        print(f"Lỗi khi xử lý file {json_path}: {e}")
        return None

def copy_and_process_data(matched_pairs, output_dir, img_size=(512, 512), 
                          val_split=0.15, test_split=0.15, label_mapping=None):
    """Sao chép và xử lý dữ liệu sang thư mục đầu ra."""
    print("Đang xử lý và sao chép dữ liệu...")
    
    # Chia dữ liệu thành train, val, test
    random.shuffle(matched_pairs)
    n_total = len(matched_pairs)
    n_test = int(n_total * test_split)
    n_val = int(n_total * val_split)
    n_train = n_total - n_test - n_val
    
    train_pairs = matched_pairs[:n_train]
    val_pairs = matched_pairs[n_train:n_train+n_val]
    test_pairs = matched_pairs[n_train+n_val:]
    
    print(f"Chia dữ liệu: {n_train} train, {n_val} validation, {n_test} test")
    
    # Xử lý từng phần
    process_subset(train_pairs, output_dir, 'train', img_size, label_mapping)
    process_subset(val_pairs, output_dir, 'val', img_size, label_mapping)
    process_subset(test_pairs, output_dir, 'test', img_size, label_mapping)
    
    # Lưu tất cả vào thư mục raw để tham khảo
    raw_dir = os.path.join(output_dir, 'raw')
    for img_path, json_path in tqdm(matched_pairs, desc="Sao chép raw data"):
        # Sao chép file ảnh
        img_name = os.path.basename(img_path)
        shutil.copy2(img_path, os.path.join(raw_dir, img_name))
        
        # Sao chép file json
        json_name = os.path.basename(json_path)
        shutil.copy2(json_path, os.path.join(raw_dir, json_name))
    
    print("Hoàn thành xử lý dữ liệu!")

def process_subset(pairs, output_dir, subset, img_size, label_mapping):
    """Xử lý một tập con dữ liệu (train/val/test)."""
    images_dir = os.path.join(output_dir, f'segmentation/{subset}/images')
    masks_dir = os.path.join(output_dir, f'segmentation/{subset}/masks')
    
    # Xử lý song song để tăng tốc
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for img_path, json_path in pairs:
            future = executor.submit(
                process_single_pair, 
                img_path, 
                json_path, 
                images_dir, 
                masks_dir, 
                img_size, 
                label_mapping
            )
            futures.append(future)
        
        # Hiển thị tiến trình
        for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Xử lý {subset}"):
            pass

def process_single_pair(img_path, json_path, images_dir, masks_dir, img_size, label_mapping):
    """Xử lý một cặp file ảnh và annotation."""
    try:
        # Lấy tên file gốc
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        
        # Đọc và resize ảnh
        img = cv2.imread(img_path)
        if img is None:
            print(f"Không thể đọc ảnh {img_path}")
            return
        
        img_resized = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        
        # Tạo mask từ file json
        mask = process_json_to_mask(json_path, img_size, label_mapping)
        if mask is None:
            print(f"Không thể tạo mask từ {json_path}")
            return
        
        # Lưu ảnh và mask
        cv2.imwrite(os.path.join(images_dir, f"{base_name}.jpg"), img_resized)
        cv2.imwrite(os.path.join(masks_dir, f"{base_name}.png"), mask)
        
    except Exception as e:
        print(f"Lỗi khi xử lý {img_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Chuẩn bị dữ liệu phân đoạn xoài')
    parser.add_argument('--input_dir', type=str, required=True, help='Thư mục chứa dữ liệu gốc')
    parser.add_argument('--output_dir', type=str, default='data', help='Thư mục đầu ra')
    parser.add_argument('--img_size', type=int, nargs=2, default=[512, 512], help='Kích thước ảnh output (width, height)')
    parser.add_argument('--val_split', type=float, default=0.15, help='Tỷ lệ chia cho validation')
    parser.add_argument('--test_split', type=float, default=0.15, help='Tỷ lệ chia cho test')
    
    args = parser.parse_args()
    
    # Thiết lập mapping cho các nhãn bệnh trên xoài
    label_mapping = {
        "DC": 1,  # Da cám
        "DE": 2,  # Da ếch
        "DD": 3,  # Đóm đen
        "TT": 4,  # Thán thư
        "RD": 5,  # Rùi đụt
    }
    
    # Thiết lập cấu trúc thư mục
    setup_directory_structure(args.output_dir)
    
    # Thu thập và ghép cặp file
    image_files, json_files = collect_data_files(args.input_dir)
    matched_pairs = match_image_annotation(image_files, json_files)
    
    # Xử lý và sao chép dữ liệu
    copy_and_process_data(
        matched_pairs, 
        args.output_dir, 
        img_size=tuple(args.img_size), 
        val_split=args.val_split, 
        test_split=args.test_split,
        label_mapping=label_mapping
    )
    
    print("Hoàn thành chuẩn bị dữ liệu!")
    print(f"Dữ liệu đã được lưu trong thư mục {args.output_dir}")

if __name__ == "__main__":
    main()