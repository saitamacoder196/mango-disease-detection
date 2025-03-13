import os
import cv2
import numpy as np
import json
from tqdm import tqdm
import glob
import argparse

def process_json_to_mask(json_path, output_size=(512, 512)):
    """Chuyển đổi file JSON annotation thành mask."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Lấy kích thước ảnh gốc
        img_height = data.get('imageHeight', output_size[0])
        img_width = data.get('imageWidth', output_size[1])
        
        # Tạo mask trống
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        
        # Mapping cho các nhãn bệnh
        label_mapping = {
            "DC": 1,  # Da cám
            "DE": 2,  # Da ếch
            "DD": 3,  # Đóm đen
            "TT": 4,  # Thán thư
            "RD": 5,  # Rùi đụt
        }
        
        # Vẽ các đa giác lên mask
        for shape in data.get('shapes', []):
            label = shape.get('label')
            points = shape.get('points')
            
            # Bỏ qua nếu không có đủ điểm để tạo polygon
            if not points or len(points) < 3:
                continue
                
            # Chuyển đổi label thành ID nếu có mapping
            label_id = label_mapping.get(label, 1)
            
            # Chuyển đổi points thành định dạng phù hợp cho cv2.fillPoly
            points_array = np.array(points, dtype=np.int32)
            
            # Vẽ polygon
            cv2.fillPoly(mask, [points_array], label_id)
        
        # Resize mask về kích thước mong muốn
        if output_size and (img_height != output_size[0] or img_width != output_size[1]):
            mask = cv2.resize(mask, output_size, interpolation=cv2.INTER_NEAREST)
        
        # Kiểm tra nếu mask toàn đen
        if np.max(mask) == 0:
            print(f"WARNING: Mask cho {os.path.basename(json_path)} toàn đen!")
        
        return mask
    
    except Exception as e:
        print(f"Lỗi khi xử lý file {json_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def fix_masks_from_json(raw_dir, masks_dir, img_size=(512, 512)):
    """Sửa các mask bằng cách tạo lại từ file JSON."""
    print(f"Đang sửa chữa masks từ JSON trong {raw_dir}...")
    
    # Tìm tất cả file JSON trong thư mục raw
    json_files = glob.glob(os.path.join(raw_dir, '*.json'))
    print(f"Tìm thấy {len(json_files)} file JSON")
    
    # Tạo thư mục masks nếu chưa tồn tại
    os.makedirs(masks_dir, exist_ok=True)
    
    # Tạo lại mask từ từng file JSON
    for json_path in tqdm(json_files):
        # Lấy tên file gốc
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        
        # Tạo mask từ file JSON
        mask = process_json_to_mask(json_path, img_size)
        if mask is None:
            continue
        
        # Lưu mask
        mask_path = os.path.join(masks_dir, f"{base_name}.png")
        cv2.imwrite(mask_path, mask)
        
        # Tạo phiên bản mask màu để kiểm tra trực quan
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        
        # Màu cho các loại bệnh (BGR)
        colors = [
            [0, 0, 0],        # Background - đen
            [0, 0, 255],      # DC - đỏ
            [0, 255, 0],      # DE - xanh lá
            [255, 0, 0],      # DD - xanh dương
            [0, 255, 255],    # TT - vàng
            [255, 0, 255]     # RD - tím
        ]
        
        # Tô màu mask
        for class_idx, color in enumerate(colors):
            colored_mask[mask == class_idx] = color
        
        # Lưu mask màu
        colored_mask_path = os.path.join(masks_dir, f"{base_name}_colored.png")
        cv2.imwrite(colored_mask_path, colored_mask)

def check_and_fix_dataset(data_dir, img_size=(512, 512)):
    """Kiểm tra và sửa chữa toàn bộ dataset."""
    # Thư mục raw chứa JSON gốc
    raw_dir = os.path.join(data_dir, "raw")
    
    # Các thư mục cần sửa chữa
    for subset in ["train", "val", "test"]:
        print(f"\nĐang xử lý tập {subset}...")
        
        # Thư mục chứa mask cần sửa
        masks_dir = os.path.join(data_dir, f"segmentation/{subset}/masks")
        
        # Tạo thư mục mới để lưu mask đã sửa
        fixed_masks_dir = os.path.join(data_dir, f"segmentation/{subset}/fixed_masks")
        os.makedirs(fixed_masks_dir, exist_ok=True)
        
        # Kiểm tra mask hiện tại có phải toàn đen không
        mask_files = glob.glob(os.path.join(masks_dir, "*.png"))
        all_black = True
        
        for mask_path in tqdm(mask_files, desc="Kiểm tra mask hiện tại"):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if np.max(mask) > 0:
                all_black = False
                break
        
        if all_black:
            print(f"Tất cả mask trong {masks_dir} đều toàn đen. Đang sửa chữa...")
            
            # Tạo lại mask từ JSON
            fix_masks_from_json(raw_dir, fixed_masks_dir, img_size)
            
            # Thay thế thư mục mask cũ bằng thư mục đã sửa
            os.rename(masks_dir, masks_dir + "_old")
            os.rename(fixed_masks_dir, masks_dir)
            print(f"Đã thay thế mask cũ bằng mask đã sửa!")
        else:
            print(f"Mask trong {masks_dir} không toàn đen, không cần sửa chữa.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Kiểm tra và sửa chữa mask trong dataset')
    parser.add_argument('--data_dir', type=str, default='data', help='Thư mục chứa dataset')
    parser.add_argument('--img_size', type=int, nargs=2, default=[512, 512], help='Kích thước ảnh (width, height)')
    
    args = parser.parse_args()
    
    check_and_fix_dataset(args.data_dir, tuple(args.img_size))