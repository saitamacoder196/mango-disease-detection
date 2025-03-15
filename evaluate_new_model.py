# evaluate_new_model.py
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import yaml
from tensorflow.keras.models import load_model

# Định nghĩa các hàm metrics tùy chỉnh giống như trong huấn luyện
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

# Màu cho các lớp (BGR cho OpenCV)
CLASS_NAMES = ["background", "da_cam", "da_ech", "dom_den", "than_thu", "rui_dut"]
COLORS = [
    [0, 0, 0],      # Background - đen
    [0, 0, 255],    # Da cám - đỏ
    [0, 255, 0],    # Da ếch - xanh lá
    [255, 0, 0],    # Đóm đen - xanh dương
    [0, 255, 255],  # Thán thư - vàng
    [255, 0, 255]   # Rùi đụt - tím
]

# Tạo mask màu
def create_colored_mask(mask):
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_idx, color in enumerate(COLORS):
        colored_mask[mask == class_idx] = color
    return colored_mask

# Đọc cấu hình
with open('configs/segmentation_config_new.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Tải mô hình mới
model_path = 'models/weighted_segmentation_model.h5'
print(f"Đang tải mô hình từ {model_path}...")
model = load_model(
    model_path,
    custom_objects={
        'iou_score': iou_score,
        'f1_score': f1_score
    }
)
print("Đã tải mô hình thành công")

# Đánh giá trên tập test
test_dir = config['data']['test_dir']
img_dir = os.path.join(test_dir, 'images')
mask_dir = os.path.join(test_dir, 'masks')

# Lấy danh sách ảnh
img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
print(f"Tìm thấy {len(img_files)} ảnh trong tập test")

# Tạo thư mục đầu ra
output_dir = 'evaluation_results'
os.makedirs(output_dir, exist_ok=True)

# Chọn một số ảnh ngẫu nhiên để hiển thị
num_samples = min(5, len(img_files))
sample_indices = np.random.choice(len(img_files), num_samples, replace=False)

# Metrics
iou_scores = []
f1_scores = []
class_metrics = {i: {'tp': 0, 'fp': 0, 'fn': 0} for i in range(len(CLASS_NAMES))}

# Xử lý từng ảnh
for i, img_file in enumerate(img_files):
    # Đọc ảnh
    img_path = os.path.join(img_dir, img_file)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize ảnh
    img_size = tuple(config['data']['img_size'])
    img_resized = cv2.resize(img, img_size)
    
    # Đọc mask thực tế
    mask_path = os.path.join(mask_dir, os.path.splitext(img_file)[0] + '.png')
    if os.path.exists(mask_path):
        true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        true_mask = cv2.resize(true_mask, img_size, interpolation=cv2.INTER_NEAREST)
    else:
        print(f"Không tìm thấy mask tương ứng cho {img_file}")
        continue
    
    # Dự đoán
    img_input = img_resized / 255.0
    img_input = np.expand_dims(img_input, axis=0)
    
    pred = model.predict(img_input)[0]
    pred_mask = np.argmax(pred, axis=-1)
    
    # Tính metrics
    accuracy = np.mean(pred_mask == true_mask)
    
    # Tính IoU và F1 cho từng lớp
    for class_idx in range(len(CLASS_NAMES)):
        true_binary = (true_mask == class_idx)
        pred_binary = (pred_mask == class_idx)
        
        # True positives, false positives, false negatives
        tp = np.sum(np.logical_and(pred_binary, true_binary))
        fp = np.sum(np.logical_and(pred_binary, np.logical_not(true_binary)))
        fn = np.sum(np.logical_and(np.logical_not(pred_binary), true_binary))
        
        class_metrics[class_idx]['tp'] += tp
        class_metrics[class_idx]['fp'] += fp
        class_metrics[class_idx]['fn'] += fn
    
    # Hiển thị kết quả cho ảnh mẫu
    if i in sample_indices:
        plt.figure(figsize=(15, 5))
        
        # Ảnh gốc
        plt.subplot(1, 3, 1)
        plt.imshow(img_resized)
        plt.title("Ảnh gốc")
        plt.axis('off')
        
        # Mask thực tế
        true_colored = create_colored_mask(true_mask)
        plt.subplot(1, 3, 2)
        plt.imshow(true_colored)
        plt.title("Mask thực tế")
        plt.axis('off')
        
        # Mask dự đoán
        pred_colored = create_colored_mask(pred_mask)
        plt.subplot(1, 3, 3)
        plt.imshow(pred_colored)
        plt.title("Mask dự đoán")
        plt.axis('off')
        
        plt.suptitle(f"Ảnh: {img_file}, Pixel Accuracy: {accuracy:.4f}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"sample_{i}_{img_file}.png"))
        plt.close()

# Tính precision, recall, F1, IoU cho từng lớp
results = []
for class_idx in range(len(CLASS_NAMES)):
    tp = class_metrics[class_idx]['tp']
    fp = class_metrics[class_idx]['fp']
    fn = class_metrics[class_idx]['fn']
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    results.append({
        'class': CLASS_NAMES[class_idx],
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou
    })

# Hiển thị kết quả
print("\nKết quả đánh giá trên tập test:")
print(f"{'Lớp':<15} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'IoU':<10}")
print("-" * 60)

for result in results:
    print(f"{result['class']:<15} {result['precision']:.4f}{'':6} {result['recall']:.4f}{'':6} {result['f1']:.4f}{'':6} {result['iou']:.4f}")

# Tính trung bình (không tính background)
avg_precision = np.mean([r['precision'] for r in results[1:]])
avg_recall = np.mean([r['recall'] for r in results[1:]])
avg_f1 = np.mean([r['f1'] for r in results[1:]])
avg_iou = np.mean([r['iou'] for r in results[1:]])

print("-" * 60)
print(f"{'Trung bình':<15} {avg_precision:.4f}{'':6} {avg_recall:.4f}{'':6} {avg_f1:.4f}{'':6} {avg_iou:.4f}")

# Lưu kết quả
with open(os.path.join(output_dir, 'evaluation_results.txt'), 'w') as f:
    f.write("Kết quả đánh giá trên tập test:\n")
    f.write(f"{'Lớp':<15} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'IoU':<10}\n")
    f.write("-" * 60 + "\n")
    
    for result in results:
        f.write(f"{result['class']:<15} {result['precision']:.4f}{'':6} {result['recall']:.4f}{'':6} {result['f1']:.4f}{'':6} {result['iou']:.4f}\n")
    
    f.write("-" * 60 + "\n")
    f.write(f"{'Trung bình':<15} {avg_precision:.4f}{'':6} {avg_recall:.4f}{'':6} {avg_f1:.4f}{'':6} {avg_iou:.4f}\n")

print(f"\nĐã lưu kết quả đánh giá vào {os.path.join(output_dir, 'evaluation_results.txt')}")
print(f"Các ảnh mẫu đã được lưu trong thư mục {output_dir}")