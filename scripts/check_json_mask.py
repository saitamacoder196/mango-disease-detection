import json
import numpy as np
import cv2
import os

# Đường dẫn đến file JSON của bạn
json_path = "D:\\05_Mango_Segmentation_Data\\Dot1\\DC\\1-XL-DC-Center_1.json"  # Thay thế bằng đường dẫn thực tế

# Mapping cho các nhãn
LABEL_MAPPING = {
    "background": 0,
    "DC": 1,  # Da cám
    "DE": 2,  # Da ếch
    "DD": 3,  # Đóm đen
    "TT": 4,  # Thán thư
    "RD": 5,  # Rùi đụt
}

# Màu cho hiển thị (BGR cho OpenCV)
COLORS = [
    [0, 0, 0],        # Background - đen
    [0, 0, 255],      # DC - đỏ
    [0, 255, 0],      # DE - xanh lá
    [255, 0, 0],      # DD - xanh dương
    [0, 255, 255],    # TT - vàng
    [255, 0, 255]     # RD - tím
]

def create_mask_from_json(json_path, output_dir):
    # Đọc file JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Lấy kích thước ảnh gốc
    img_height = data.get('imageHeight')
    img_width = data.get('imageWidth')
    
    print(f"Kích thước ảnh gốc: {img_width}x{img_height}")
    
    # Tạo mask với kích thước gốc
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    # Vẽ từng shape
    for i, shape in enumerate(data.get('shapes', [])):
        label = shape.get('label')
        points = shape.get('points')
        
        print(f"Xử lý shape {i+1}/{len(data.get('shapes', []))}: label={label}, số điểm={len(points)}")
        
        # Kiểm tra xem label có trong mapping không
        if label not in LABEL_MAPPING:
            print(f"WARNING: Label '{label}' không có trong mapping!")
            label_id = 1  # Gán giá trị mặc định
        else:
            label_id = LABEL_MAPPING[label]
        
        # Chuyển points thành numpy array
        points_array = np.array(points, dtype=np.int32)
        
        # In một số điểm đầu tiên để kiểm tra
        print(f"Mẫu điểm: {points_array[:3] if len(points_array) >= 3 else points_array}")
        
        # Vẽ polygon
        cv2.fillPoly(mask, [points_array], label_id)
        
        # Đếm số pixel để kiểm tra
        pixel_count = np.sum(mask == label_id)
        print(f"Số pixel cho label {label}: {pixel_count}")
    
    # Lưu mask gốc
    original_mask_path = os.path.join(output_dir, 'original_mask.png')
    cv2.imwrite(original_mask_path, mask * 50)  # Nhân với 50 để dễ nhìn
    print(f"Đã lưu mask gốc tại: {original_mask_path}")
    
    # Tạo mask màu để trực quan hóa
    colored_mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    for class_idx, color in enumerate(COLORS):
        colored_mask[mask == class_idx] = color
    
    # Lưu mask màu
    colored_mask_path = os.path.join(output_dir, 'colored_mask.png')
    cv2.imwrite(colored_mask_path, colored_mask)
    print(f"Đã lưu mask màu tại: {colored_mask_path}")
    
    # Tạo các phiên bản resize
    resize_sizes = [(512, 512), (256, 256)]
    for size in resize_sizes:
        # Resize mask
        resized_mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        
        # Lưu mask đã resize
        resized_path = os.path.join(output_dir, f'mask_{size[0]}x{size[1]}.png')
        cv2.imwrite(resized_path, resized_mask * 50)
        print(f"Đã lưu mask resize {size[0]}x{size[1]} tại: {resized_path}")
        
        # Kiểm tra giá trị sau khi resize
        unique_resized = np.unique(resized_mask)
        print(f"Các giá trị trong mask sau khi resize {size[0]}x{size[1]}: {unique_resized}")
    
    return mask

def check_mask_distribution(mask_dir):
    """Kiểm tra phân phối lớp trong các file mask."""
    mask_files = glob.glob(os.path.join(mask_dir, "*.png"))
    class_counts = {i: 0 for i in range(6)}  # Lớp 0-5
    total_pixels = 0
    
    for mask_path in mask_files:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        
        total_pixels += mask.size
        for class_idx in range(6):
            class_counts[class_idx] += np.sum(mask == class_idx)
    
    print("Phân phối lớp trong mask:")
    for class_idx, count in class_counts.items():
        percentage = (count / total_pixels) * 100 if total_pixels > 0 else 0
        print(f"Lớp {class_idx} ({CLASS_NAMES[class_idx]}): {percentage:.2f}%")

# Gọi hàm để kiểm tra
check_mask_distribution('data/segmentation/train/masks')

# # Tạo thư mục đầu ra
# output_dir = 'debug_output'  # Có thể thay đổi thành thư mục khác
# os.makedirs(output_dir, exist_ok=True)

# # Tạo mask
# mask = create_mask_from_json(json_path, output_dir)

# print("Hoàn thành!")