# test_new_model.py
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import argparse
from tensorflow.keras.models import load_model

# Định nghĩa hàm metrics
def iou_score(y_true, y_pred, threshold=0.5, smooth=1e-5):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    pred_positive = tf.keras.backend.cast(tf.keras.backend.greater_equal(y_pred_f, threshold), 'float32')
    intersection = tf.keras.backend.sum(y_true_f * pred_positive)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(pred_positive) - intersection
    return (intersection + smooth) / (union + smooth)

def f1_score(y_true, y_pred, threshold=0.5, smooth=1e-5):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    pred_positive = tf.keras.backend.cast(tf.keras.backend.greater_equal(y_pred_f, threshold), 'float32')
    true_positive = tf.keras.backend.sum(y_true_f * pred_positive)
    precision = true_positive / (tf.keras.backend.sum(pred_positive) + smooth)
    recall = true_positive / (tf.keras.backend.sum(y_true_f) + smooth)
    return 2 * precision * recall / (precision + recall + smooth)

# Tên các lớp
CLASS_NAMES = ["background", "da_cam", "da_ech", "dom_den", "than_thu", "rui_dut"]

# Màu cho các lớp (RGB)
COLORS = [
    [0, 0, 0],      # Background - đen
    [255, 0, 0],    # Da cám - đỏ
    [0, 255, 0],    # Da ếch - xanh lá
    [0, 0, 255],    # Đóm đen - xanh dương
    [255, 255, 0],  # Thán thư - vàng
    [255, 0, 255]   # Rùi đụt - tím
]

def create_colored_mask(mask):
    """Tạo mask màu từ mask grayscale."""
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_idx, color in enumerate(COLORS):
        colored_mask[mask == class_idx] = color
    return colored_mask

def main():
    parser = argparse.ArgumentParser(description='Thử nghiệm mô hình phân đoạn trên ảnh')
    parser.add_argument('--model', type=str, default='models/weighted_segmentation_model.h5',
                        help='Đường dẫn đến mô hình')
    parser.add_argument('--image', type=str, required=True,
                        help='Đường dẫn đến ảnh cần phân tích')
    parser.add_argument('--output', type=str, default='result.png',
                        help='Đường dẫn lưu kết quả')
    
    args = parser.parse_args()
    
    # Tải mô hình
    print(f"Đang tải mô hình từ {args.model}...")
    model = load_model(
        args.model,
        custom_objects={
            'iou_score': iou_score,
            'f1_score': f1_score
        }
    )
    print("Đã tải mô hình thành công")
    
    # Đọc ảnh
    img = cv2.imread(args.image)
    if img is None:
        print(f"Không thể đọc ảnh từ {args.image}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize ảnh
    img_size = (512, 512)  # Kích thước chuẩn
    img_resized = cv2.resize(img, img_size)
    
    # Dự đoán
    img_input = img_resized / 255.0
    img_input = np.expand_dims(img_input, axis=0)
    
    pred = model.predict(img_input)[0]
    pred_mask = np.argmax(pred, axis=-1)
    
    # Tạo mask màu
    colored_mask = create_colored_mask(pred_mask)
    
    # Tạo overlay
    alpha = 0.6
    overlay = cv2.addWeighted(img_resized, 1-alpha, colored_mask, alpha, 0)
    
    # Tính phần trăm diện tích
    total_pixels = pred_mask.size
    class_areas = {}
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        pixel_count = np.sum(pred_mask == class_idx)
        percentage = (pixel_count / total_pixels) * 100
        class_areas[class_name] = percentage
    
    # Sắp xếp theo diện tích giảm dần
    sorted_areas = sorted(class_areas.items(), key=lambda x: x[1], reverse=True)
    
    # Hiển thị kết quả
    plt.figure(figsize=(15, 10))
    
    # Ảnh gốc
    plt.subplot(2, 2, 1)
    plt.imshow(img_resized)
    plt.title("Ảnh gốc")
    plt.axis('off')
    
    # Mask dự đoán
    plt.subplot(2, 2, 2)
    plt.imshow(colored_mask)
    plt.title("Mask phân đoạn")
    plt.axis('off')
    
    # Overlay
    plt.subplot(2, 2, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis('off')
    
    # Phần trăm diện tích
    plt.subplot(2, 2, 4)
    plt.axis('off')
    plt.title("Phần trăm diện tích")
    
    # Hiển thị phần trăm diện tích bằng biểu đồ ngang
    filtered_areas = [(name, pct) for name, pct in sorted_areas if pct > 0]
    
    if filtered_areas:
        names = [name for name, _ in filtered_areas]
        percentages = [pct for _, pct in filtered_areas]
        colors = [COLORS[CLASS_NAMES.index(name)] for name, _ in filtered_areas]
        # Chuyển từ RGB sang định dạng màu của matplotlib
        colors = [[r/255, g/255, b/255] for r, g, b in colors]
        
        y_pos = np.arange(len(names))
        plt.barh(y_pos, percentages, color=colors)
        plt.yticks(y_pos, names)
        for i, v in enumerate(percentages):
            if v > 0.01:  # Chỉ hiển thị giá trị > 0.01%
                plt.text(v + 0.5, i, f"{v:.2f}%", va='center')
    
    plt.tight_layout()
    plt.savefig(args.output)
    
    # In kết quả
    print("\nPhần trăm diện tích từng loại bệnh:")
    for class_name, percentage in sorted_areas:
        if percentage > 0.01:  # Chỉ hiển thị giá trị > 0.01%
            print(f"{class_name}: {percentage:.2f}%")
    
    print(f"\nĐã lưu kết quả vào {args.output}")
    
    # Hiển thị ảnh nếu có thể
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    main()