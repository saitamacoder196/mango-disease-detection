# File: detailed_check.py

import os
import cv2
import numpy as np
import glob
from tqdm import tqdm

def check_masks_in_detail(masks_dir):
    """Kiểm tra chi tiết các mask trong thư mục."""
    print(f"Kiểm tra chi tiết masks trong {masks_dir}...")
    
    # Lấy danh sách tất cả file mask
    mask_files = glob.glob(os.path.join(masks_dir, "*.png"))
    print(f"Tìm thấy {len(mask_files)} file mask")
    
    all_black = True
    nonzero_masks = []
    
    for mask_path in tqdm(mask_files, desc="Kiểm tra chi tiết"):
        # Đọc mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Kiểm tra giá trị pixel và độ sáng
        min_val = np.min(mask)
        max_val = np.max(mask)
        mean_val = np.mean(mask)
        nonzero = np.count_nonzero(mask)
        
        # Nếu có pixel khác 0
        if max_val > 0:
            all_black = False
            nonzero_masks.append(os.path.basename(mask_path))
            print(f"\nFile: {os.path.basename(mask_path)}")
            print(f"  Min value: {min_val}")
            print(f"  Max value: {max_val}")
            print(f"  Mean value: {mean_val:.2f}")
            print(f"  Non-zero pixels: {nonzero} ({nonzero/mask.size*100:.2f}%)")
            
            # Lưu mask nhị phân với giá trị tăng cường
            enhanced_mask = (mask > 0).astype(np.uint8) * 255
            enhanced_path = os.path.join(os.path.dirname(mask_path), "enhanced_" + os.path.basename(mask_path))
            cv2.imwrite(enhanced_path, enhanced_mask)
            
            # Lưu mask màu
            colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            # Màu cho các lớp (BGR)
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
                if class_idx > 0:  # Bỏ qua background
                    colored_mask[mask == class_idx] = color
            
            # Lưu mask màu
            colored_path = os.path.join(os.path.dirname(mask_path), "colored_" + os.path.basename(mask_path))
            cv2.imwrite(colored_path, colored_mask)
        
    if all_black:
        print(f"\nTất cả mask trong {masks_dir} đều toàn đen!")
    else:
        print(f"\nCó {len(nonzero_masks)} mask trong {masks_dir} có pixel khác 0:")
        for idx, mask_name in enumerate(nonzero_masks[:10]):  # Chỉ hiển thị 10 mask đầu tiên
            print(f"  {idx+1}. {mask_name}")
        if len(nonzero_masks) > 10:
            print(f"  ... và {len(nonzero_masks)-10} mask khác")

def create_test_mask():
    """Tạo và lưu một mask đơn giản để kiểm tra."""
    # Tạo một mask đơn giản có giá trị khác 0
    mask = np.zeros((512, 512), dtype=np.uint8)
    
    # Vẽ một hình chữ nhật với giá trị 1
    cv2.rectangle(mask, (100, 100), (400, 400), 1, -1)
    
    # Vẽ một hình tròn với giá trị 2
    cv2.circle(mask, (250, 250), 100, 2, -1)
    
    # Vẽ một hình tam giác với giá trị 3
    triangle_pts = np.array([[250, 50], [150, 350], [350, 350]], dtype=np.int32)
    cv2.fillPoly(mask, [triangle_pts], 3)
    
    # Lưu mask thường
    cv2.imwrite("test_mask.png", mask)
    
    # Lưu mask tăng cường (nhân với 50)
    cv2.imwrite("test_mask_enhanced.png", mask * 50)
    
    # Lưu mask màu
    colored_mask = np.zeros((512, 512, 3), dtype=np.uint8)
    colors = [
        [0, 0, 0],        # Background - đen
        [0, 0, 255],      # DC - đỏ
        [0, 255, 0],      # DE - xanh lá
        [255, 0, 0]       # DD - xanh dương
    ]
    for class_idx, color in enumerate(colors):
        colored_mask[mask == class_idx] = color
    
    cv2.imwrite("test_mask_colored.png", colored_mask)
    
    print("Đã tạo các file mask thử nghiệm: test_mask.png, test_mask_enhanced.png, test_mask_colored.png")

if __name__ == "__main__":
    # Kiểm tra chi tiết mask trong từng tập dữ liệu
    for subset in ["train", "val", "test"]:
        masks_dir = f"data/segmentation/{subset}/masks"
        check_masks_in_detail(masks_dir)
    
    # Tạo một mask đơn giản để kiểm tra
    create_test_mask()