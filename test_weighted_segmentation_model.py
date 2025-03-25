# test_weighted_segmentation_model.py

import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import segmentation_models as sm
from tensorflow.keras.models import load_model
import glob

# Định nghĩa các thông số
IMG_SIZE = (512, 512)
CLASS_NAMES = ["background", "da_cam", "da_ech", "dom_den", "than_thu", "rui_dut"]
COLORS = [
    [0, 0, 0],      # Background - đen
    [255, 0, 0],    # Da cám - đỏ
    [0, 255, 0],    # Da ếch - xanh lá
    [0, 0, 255],    # Đóm đen - xanh dương
    [255, 255, 0],  # Thán thư - vàng
    [255, 0, 255]   # Rùi đụt - tím
]
MODEL_PATH = "models/weighted_segmentation_model.h5"
TEST_DIR = "data/segmentation/test/images"
MASK_DIR = "data/segmentation/test/masks"  # Thư mục chứa mask thực tế (ground truth)
RESULTS_DIR = "test_results"

# Định nghĩa lại hàm loss đã sử dụng trong huấn luyện
def weighted_loss(y_true, y_pred):
    # Đây là một ví dụ, bạn cần điều chỉnh phù hợp với hàm loss thực tế đã sử dụng
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)

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

def create_colored_mask(mask):
    """Tạo mask màu từ mask grayscale."""
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_idx, color in enumerate(COLORS):
        colored_mask[mask == class_idx] = color
    return colored_mask

def create_enhanced_mask(mask):
    """Tạo mask tăng cường để dễ nhìn hơn."""
    # Nhân với 50 (hoặc giá trị khác) để làm nổi bật giá trị
    enhanced_mask = mask * 50
    return enhanced_mask

def create_comparison_mask(pred_mask, true_mask=None):
    """
    Tạo mask so sánh giữa vùng thực tế khác 0 và vùng dự đoán khác 0.
    - Vùng thực tế (ground truth) khác 0 được tô màu xanh lá (0, 255, 0)
    - Vùng dự đoán khác 0 được tô màu đỏ (255, 0, 0)
    """
    if true_mask is None:
        return None
    
    # Tạo mask so sánh
    comparison = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    
    # Tô màu xanh lá cho vùng thực tế khác 0
    true_nonzero = (true_mask > 0)
    comparison[true_nonzero] = [0, 255, 0]  # Màu xanh lá
    
    # Tô màu đỏ cho vùng dự đoán khác 0
    pred_nonzero = (pred_mask > 0)
    comparison[pred_nonzero] = [255, 0, 0]  # Màu đỏ
    
    # Các pixel đồng thời thuộc cả hai vùng sẽ bị ghi đè bởi màu đỏ
    
    return comparison

def load_segmentation_model(model_path):
    """Tải mô hình phân đoạn."""
    print(f"Đang tải mô hình từ {model_path}...")
    
    try:
        # Tải mô hình với các hàm tùy chỉnh
        model = load_model(
            model_path,
            custom_objects={
                'iou_score': iou_score,
                'f1_score': f1_score,
                'f1-score': f1_score,
                'loss': weighted_loss,
                'weighted_loss': weighted_loss
            }
        )
        print("Đã tải mô hình thành công!")
        return model
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        import traceback
        traceback.print_exc()
        return None

def predict_segmentation(model, image_path, true_mask_path=None):
    """Dự đoán phân đoạn trên một ảnh."""
    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"Không thể đọc ảnh từ {image_path}")
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize ảnh
    img_resized = cv2.resize(img, IMG_SIZE)
    
    # Chuẩn bị đầu vào
    img_input = img_resized / 255.0
    img_input = np.expand_dims(img_input, axis=0)
    
    # Dự đoán
    pred = model.predict(img_input)[0]
    pred_mask = np.argmax(pred, axis=-1)
    
    # Tạo mask màu
    colored_mask = create_colored_mask(pred_mask)
    
    # Tạo mask tăng cường
    enhanced_mask = create_enhanced_mask(pred_mask)
    
    # Tạo overlay
    alpha = 0.6
    overlay_img = cv2.addWeighted(img_resized, 1-alpha, colored_mask, alpha, 0)
    
    # Tính phần trăm diện tích từng loại bệnh
    total_pixels = pred_mask.size
    class_areas = {}
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        pixel_count = np.sum(pred_mask == class_idx)
        percentage = (pixel_count / total_pixels) * 100
        class_areas[class_name] = percentage
    
    # Xử lý mask thực tế nếu có
    true_mask = None
    comparison_mask = None
    
    if true_mask_path and os.path.exists(true_mask_path):
        true_mask = cv2.imread(true_mask_path, cv2.IMREAD_GRAYSCALE)
        if true_mask is not None:
            true_mask = cv2.resize(true_mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
            comparison_mask = create_comparison_mask(pred_mask, true_mask)
            
            # Tính IoU (Intersection over Union)
            pred_nonzero = (pred_mask > 0)
            true_nonzero = (true_mask > 0)
            
            intersection = np.logical_and(pred_nonzero, true_nonzero).sum()
            union = np.logical_or(pred_nonzero, true_nonzero).sum()
            
            if union > 0:
                iou = intersection / union
                print(f"IoU (Intersection over Union): {iou:.4f}")
            else:
                print("Không có vùng được phân đoạn trong cả hai mask")
    
    return img_resized, pred_mask, colored_mask, enhanced_mask, overlay_img, class_areas, true_mask, comparison_mask

def visualize_results(img, colored_mask, enhanced_mask, overlay, comparison_mask, class_areas, save_path=None):
    """Hiển thị kết quả phân đoạn."""
    has_comparison = comparison_mask is not None
    num_plots = 5 if has_comparison else 4
    
    plt.figure(figsize=(20, 15 if has_comparison else 12))
    
    # Ảnh gốc
    plt.subplot(3 if has_comparison else 2, 2, 1)
    plt.imshow(img)
    plt.title("Ảnh gốc", fontsize=14)
    plt.axis('off')
    
    # Mask màu
    plt.subplot(3 if has_comparison else 2, 2, 2)
    plt.imshow(colored_mask)
    plt.title("Mask phân đoạn (Colored)", fontsize=14)
    plt.axis('off')
    
    # Vẽ chú thích cho các loại bệnh
    for i, (class_name, color) in enumerate(zip(CLASS_NAMES, COLORS)):
        plt.annotate(
            class_name, 
            xy=(0.02, 0.98 - i*0.08), 
            xycoords='axes fraction',
            color=[c/255 for c in color] if sum(color) > 0 else [1, 1, 1],
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc=[c/255 for c in color], alpha=0.8)
        )
    
    # Mask tăng cường
    plt.subplot(3 if has_comparison else 2, 2, 3)
    plt.imshow(enhanced_mask, cmap='viridis')
    plt.title("Mask phân đoạn (Enhanced)", fontsize=14)
    plt.axis('off')
    
    # Overlay
    plt.subplot(3 if has_comparison else 2, 2, 4)
    plt.imshow(overlay)
    plt.title("Overlay", fontsize=14)
    plt.axis('off')
    
    # So sánh nếu có
    if has_comparison:
        plt.subplot(3, 2, 5)
        plt.imshow(comparison_mask)
        plt.title("So sánh: Thực tế (xanh) và Dự đoán (đỏ)", fontsize=14)
        plt.axis('off')
        
        # Vẽ chú thích cho so sánh
        plt.annotate(
            "Thực tế (Mask > 0)", 
            xy=(0.02, 0.98), 
            xycoords='axes fraction',
            color=[0, 0.8, 0],
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc=[0, 0.8, 0], alpha=0.8)
        )
        
        plt.annotate(
            "Dự đoán (Mask > 0)", 
            xy=(0.02, 0.88), 
            xycoords='axes fraction',
            color=[0.8, 0, 0],
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc=[0.8, 0, 0], alpha=0.8)
        )
    
    # Vẽ biểu đồ phần trăm diện tích
    if has_comparison:
        plt.subplot(3, 2, 6)
    else:
        plt.figure(figsize=(10, 6))
    
    # Sắp xếp theo thứ tự giảm dần
    sorted_areas = sorted([(k, v) for k, v in class_areas.items() if v > 0], key=lambda x: x[1], reverse=True)
    
    if sorted_areas:
        class_names = [item[0] for item in sorted_areas]
        percentages = [item[1] for item in sorted_areas]
        
        # Tạo màu tương ứng
        colors = [[c/255 for c in COLORS[CLASS_NAMES.index(name)]] for name in class_names]
        
        bars = plt.barh(class_names, percentages, color=colors)
        plt.title("Phần trăm diện tích các loại bệnh", fontsize=14)
        plt.xlabel("Phần trăm (%)")
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        
        # Ghi giá trị phần trăm
        for i, (bar, percentage) in enumerate(zip(bars, percentages)):
            plt.text(
                percentage + 0.5, 
                i, 
                f"{percentage:.2f}%", 
                va='center', 
                fontsize=10
            )
    else:
        plt.text(0.5, 0.5, "Không có loại bệnh nào được phát hiện", 
                ha='center', va='center', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Đã lưu kết quả vào {save_path}")
    
    plt.close()

def main():
    # Tạo thư mục kết quả
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'colored'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'enhanced'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'overlay'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'raw_masks'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'comparison'), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, 'complete'), exist_ok=True)
    
    # Tải mô hình
    model = load_segmentation_model(MODEL_PATH)
    if model is None:
        return
    
    # Lấy danh sách ảnh test
    test_images = []
    for root, _, files in os.walk(TEST_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(os.path.join(root, file))
    
    if not test_images:
        print(f"Không tìm thấy ảnh trong thư mục {TEST_DIR}")
        return
    
    print(f"Đã tìm thấy {len(test_images)} ảnh test")
    
    # Kết quả phân tích tổng hợp
    all_results = {}
    
    # Xử lý từng ảnh
    for img_path in test_images:
        print(f"\nĐang xử lý ảnh: {os.path.basename(img_path)}")
        
        try:
            # Tìm mask thực tế tương ứng
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            true_mask_path = os.path.join(MASK_DIR, f"{base_name}.png")
            
            if not os.path.exists(true_mask_path):
                print(f"Không tìm thấy mask thực tế cho ảnh {base_name}")
                true_mask_path = None
            
            # Dự đoán
            img, pred_mask, colored_mask, enhanced_mask, overlay, class_areas, true_mask, comparison_mask = predict_segmentation(
                model, img_path, true_mask_path
            )
            
            # Lưu kết quả vào các thư mục
            colored_path = os.path.join(RESULTS_DIR, 'colored', f"{base_name}_colored.png")
            enhanced_path = os.path.join(RESULTS_DIR, 'enhanced', f"{base_name}_enhanced.png")
            overlay_path = os.path.join(RESULTS_DIR, 'overlay', f"{base_name}_overlay.png")
            raw_mask_path = os.path.join(RESULTS_DIR, 'raw_masks', f"{base_name}_mask.png")
            complete_path = os.path.join(RESULTS_DIR, 'complete', f"{base_name}_complete.png")
            
            # Lưu các mask
            cv2.imwrite(colored_path, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
            cv2.imwrite(enhanced_path, enhanced_mask)
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            cv2.imwrite(raw_mask_path, pred_mask)
            
            # Lưu mask so sánh nếu có
            if comparison_mask is not None:
                comparison_path = os.path.join(RESULTS_DIR, 'comparison', f"{base_name}_comparison.png")
                cv2.imwrite(comparison_path, cv2.cvtColor(comparison_mask, cv2.COLOR_RGB2BGR))
            
            # Lưu ảnh kết quả tổng hợp
            visualize_results(
                img, colored_mask, enhanced_mask, overlay, 
                comparison_mask, class_areas, complete_path
            )
            
            # Lưu kết quả phân tích
            all_results[os.path.basename(img_path)] = {
                'class_areas': class_areas,
                'has_ground_truth': true_mask is not None
            }
            
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {os.path.basename(img_path)}: {e}")
            import traceback
            traceback.print_exc()
    
    # Lưu kết quả phân tích vào file CSV
    import csv
    csv_path = os.path.join(RESULTS_DIR, 'analysis_results.csv')
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image', 'class', 'percentage', 'has_ground_truth']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for img_name, result in all_results.items():
            for class_name, percentage in result['class_areas'].items():
                writer.writerow({
                    'image': img_name,
                    'class': class_name,
                    'percentage': f"{percentage:.2f}",
                    'has_ground_truth': result['has_ground_truth']
                })
    
    print("\nĐã hoàn thành việc test mô hình!")
    print(f"Kết quả được lưu trong thư mục: {RESULTS_DIR}")
    print(f"Kết quả phân tích được lưu trong file: {csv_path}")

if __name__ == "__main__":
    main()