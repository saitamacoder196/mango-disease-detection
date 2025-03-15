# check_masks.py
import os
import cv2
import numpy as np
import glob

# Tên các lớp
CLASS_NAMES = ["background", "da_cam", "da_ech", "dom_den", "than_thu", "rui_dut"]

def check_mask_distribution(mask_dir):
    """Kiểm tra phân phối lớp trong các file mask."""
    print(f"Kiểm tra mask trong thư mục: {mask_dir}")
    mask_files = glob.glob(os.path.join(mask_dir, "*.png"))
    if not mask_files:
        print(f"Không tìm thấy file mask trong {mask_dir}")
        return
        
    print(f"Tìm thấy {len(mask_files)} file mask")
    
    class_counts = {i: 0 for i in range(len(CLASS_NAMES))}  # Lớp 0-(n-1)
    total_pixels = 0
    mask_with_disease = 0
    
    for mask_path in mask_files:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Không thể đọc mask: {mask_path}")
            continue
        
        total_pixels += mask.size
        has_disease = False
        
        for class_idx in range(len(CLASS_NAMES)):
            pixel_count = np.sum(mask == class_idx)
            class_counts[class_idx] += pixel_count
            if class_idx > 0 and pixel_count > 0:
                has_disease = True
        
        if has_disease:
            mask_with_disease += 1
    
    print(f"\nTổng số mask có chứa bệnh: {mask_with_disease}/{len(mask_files)} ({mask_with_disease/len(mask_files)*100:.2f}%)")
    print("\nPhân phối lớp trong mask:")
    for class_idx, count in class_counts.items():
        percentage = (count / total_pixels) * 100 if total_pixels > 0 else 0
        print(f"Lớp {class_idx} ({CLASS_NAMES[class_idx]}): {count:,} pixel ({percentage:.4f}%)")

# Kiểm tra mask trong các tập dữ liệu
check_mask_distribution('data/segmentation/train/masks')
print("\n" + "="*50 + "\n")
check_mask_distribution('data/segmentation/val/masks')
print("\n" + "="*50 + "\n")
check_mask_distribution('data/segmentation/test/masks')