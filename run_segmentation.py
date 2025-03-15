# File: run_segmentation.py
# Script thử nghiệm mô hình phân đoạn bệnh trên da xoài

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import yaml
import glob
import argparse
from tensorflow.keras.models import load_model
import segmentation_models as sm
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Màu cho các lớp (RGB)
CLASS_NAMES = ["background", "da_cam", "da_ech", "dom_den", "than_thu", "rui_dut"]
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

def create_enhanced_mask(mask):
    """Tạo mask tăng cường (nhân với 50)."""
    return mask * 50

def load_config(config_path):
    """Đọc file cấu hình."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_segmentation_model(model_path):
    """Tải mô hình phân đoạn."""
    print(f"Đang tải mô hình từ {model_path}...")
    try:
        # Định nghĩa metrics tương thích
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
        
        model = load_model(
            model_path,
            custom_objects={
                'iou_score': iou_score,
                'f1_score': f1_score,
                'f1-score': f1_score,
                'iou-score': iou_score,
                'IOUScore': iou_score
            }
        )
        print("Đã tải mô hình thành công")
        return model
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def predict_segmentation(model, image_path, img_size=(512, 512)):
    """
    Dự đoán phân đoạn cho một ảnh.
    
    Args:
        model: Mô hình đã huấn luyện
        image_path: Đường dẫn đến ảnh cần dự đoán
        img_size: Kích thước ảnh đầu vào
        
    Returns:
        img_resized: Ảnh gốc đã resize
        pred_mask: Mask dự đoán
        colored_mask: Mask màu
        enhanced_mask: Mask tăng cường
        overlay_img: Ảnh overlay
        class_areas: Phần trăm diện tích từng loại bệnh
    """
    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh từ {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize ảnh
    img_resized = cv2.resize(img, img_size)
    
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
    
    return img_resized, pred_mask, colored_mask, enhanced_mask, overlay_img, class_areas

def evaluate_on_test_set(model, test_dir, mask_dir, img_size=(512, 512), num_classes=6):
    """
    Đánh giá mô hình trên tập test.
    
    Args:
        model: Mô hình đã huấn luyện
        test_dir: Thư mục chứa ảnh test
        mask_dir: Thư mục chứa mask thực tế
        img_size: Kích thước ảnh đầu vào
        num_classes: Số lớp phân đoạn
        
    Returns:
        metrics_per_class: Dict chứa các metrics cho từng lớp
        avg_metrics: Dict chứa các metrics trung bình
    """
    # Lấy danh sách file ảnh
    image_files = sorted(glob.glob(os.path.join(test_dir, '*.jpg')) + 
                        glob.glob(os.path.join(test_dir, '*.jpeg')) + 
                        glob.glob(os.path.join(test_dir, '*.png')))
    
    # Khởi tạo metrics
    class_iou = {class_name: [] for class_name in CLASS_NAMES}
    class_dice = {class_name: [] for class_name in CLASS_NAMES}
    pixel_acc = []
    
    print(f"Đánh giá trên {len(image_files)} ảnh test...")
    
    # Xử lý từng ảnh
    for image_path in image_files:
        # Lấy tên file
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = os.path.join(mask_dir, f"{base_name}.png")
        
        # Kiểm tra xem mask có tồn tại không
        if not os.path.exists(mask_path):
            print(f"Không tìm thấy mask cho ảnh {base_name}")
            continue
        
        # Đọc ảnh và mask
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        
        true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        true_mask = cv2.resize(true_mask, img_size, interpolation=cv2.INTER_NEAREST)
        
        # Dự đoán
        img_input = img / 255.0
        img_input = np.expand_dims(img_input, axis=0)
        
        pred = model.predict(img_input)[0]
        pred_mask = np.argmax(pred, axis=-1)
        
        # Tính pixel accuracy
        accuracy = np.mean(pred_mask == true_mask)
        pixel_acc.append(accuracy)
        
        # Tính IoU và Dice cho từng lớp
        for class_idx, class_name in enumerate(CLASS_NAMES):
            # Tạo mask nhị phân cho lớp
            true_binary = (true_mask == class_idx).astype(np.uint8)
            pred_binary = (pred_mask == class_idx).astype(np.uint8)
            
            # Tính intersection và union
            intersection = np.logical_and(true_binary, pred_binary).sum()
            union = np.logical_or(true_binary, pred_binary).sum()
            
            # IoU
            iou = intersection / union if union > 0 else 0
            class_iou[class_name].append(iou)
            
            # Dice
            dice = 2 * intersection / (true_binary.sum() + pred_binary.sum()) if (true_binary.sum() + pred_binary.sum()) > 0 else 0
            class_dice[class_name].append(dice)
    
    # Tính trung bình cho các metrics
    avg_iou = {class_name: np.mean(scores) if scores else 0 for class_name, scores in class_iou.items()}
    avg_dice = {class_name: np.mean(scores) if scores else 0 for class_name, scores in class_dice.items()}
    avg_pixel_acc = np.mean(pixel_acc) if pixel_acc else 0
    
    # Tính trung bình tổng thể
    mean_iou = np.mean([iou for iou in avg_iou.values() if iou > 0])
    mean_dice = np.mean([dice for dice in avg_dice.values() if dice > 0])
    
    # Đóng gói kết quả
    metrics_per_class = {
        'iou': avg_iou,
        'dice': avg_dice
    }
    
    avg_metrics = {
        'mean_iou': mean_iou,
        'mean_dice': mean_dice,
        'pixel_accuracy': avg_pixel_acc
    }
    
    return metrics_per_class, avg_metrics

def display_results(img, pred_mask, colored_mask, enhanced_mask, overlay, class_areas, output_dir=None, filename=None):
    """Hiển thị và lưu kết quả phân đoạn."""
    # Hiển thị kết quả
    plt.figure(figsize=(15, 10))
    
    # Ảnh gốc
    plt.subplot(2, 3, 1)
    plt.imshow(img)
    plt.title("Ảnh gốc")
    plt.axis('off')
    
    # Mask dự đoán
    plt.subplot(2, 3, 2)
    plt.imshow(colored_mask)
    plt.title("Mask phân đoạn")
    plt.axis('off')
    
    # Mask tăng cường
    plt.subplot(2, 3, 3)
    plt.imshow(enhanced_mask, cmap='gray')
    plt.title("Mask tăng cường")
    plt.axis('off')
    
    # Overlay
    plt.subplot(2, 3, 4)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis('off')
    
    # Phần trăm diện tích
    plt.subplot(2, 3, 5)
    plt.axis('off')
    plt.title("Phần trăm diện tích")
    
    # Hiển thị phần trăm diện tích bằng biểu đồ ngang
    # Sắp xếp theo thứ tự giảm dần
    sorted_areas = sorted(class_areas.items(), key=lambda x: x[1], reverse=True)
    
    # Lọc các lớp có diện tích > 0
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
            plt.text(v + 0.5, i, f"{v:.2f}%", va='center')
    
    plt.tight_layout()
    title = "Phân đoạn bệnh trên da xoài"
    if filename:
        title += f" - {filename}"
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.9)
    
    # Lưu kết quả nếu cần
    if output_dir and filename:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"result_{os.path.splitext(filename)[0]}.png")
        plt.savefig(output_path)
        print(f"Đã lưu kết quả vào {output_path}")
    
    plt.show()
    
    # In phần trăm diện tích cho từng loại bệnh
    print("\nPhần trăm diện tích từng loại bệnh:")
    for class_name, percentage in sorted_areas:
        if percentage > 0:
            print(f"{class_name}: {percentage:.2f}%")
    print("-" * 50)

def save_results_to_csv(results, output_dir, filename="segmentation_results.csv"):
    """Lưu kết quả phân đoạn vào file CSV."""
    import csv
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image', 'class', 'percentage']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for img_name, class_areas in results.items():
            for class_name, percentage in class_areas.items():
                writer.writerow({
                    'image': img_name,
                    'class': class_name,
                    'percentage': f"{percentage:.2f}"
                })
    
    print(f"Đã lưu kết quả vào {output_path}")

def batch_process_images(model, input_dir, output_dir, img_size=(512, 512)):
    """Xử lý hàng loạt ảnh và lưu kết quả."""
    # Lấy danh sách file ảnh
    image_files = sorted(glob.glob(os.path.join(input_dir, '*.jpg')) + 
                        glob.glob(os.path.join(input_dir, '*.jpeg')) + 
                        glob.glob(os.path.join(input_dir, '*.png')))
    
    if not image_files:
        print(f"Không tìm thấy ảnh trong thư mục {input_dir}")
        return
    
    print(f"Xử lý {len(image_files)} ảnh...")
    
    # Tạo thư mục đầu ra
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "colored"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "enhanced"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "overlays"), exist_ok=True)
    
    # Lưu kết quả
    all_results = {}
    
    # Xử lý từng ảnh
    for image_path in image_files:
        try:
            filename = os.path.basename(image_path)
            print(f"Đang xử lý {filename}...")
            
            # Dự đoán
            img, pred_mask, colored_mask, enhanced_mask, overlay, class_areas = predict_segmentation(
                model, image_path, img_size=img_size
            )
            
            # Lưu ảnh kết quả
            mask_path = os.path.join(output_dir, "masks", f"mask_{filename}")
            colored_path = os.path.join(output_dir, "colored", f"colored_{filename}")
            enhanced_path = os.path.join(output_dir, "enhanced", f"enhanced_{filename}")
            overlay_path = os.path.join(output_dir, "overlays", f"overlay_{filename}")
            
            # Lưu mask gốc
            cv2.imwrite(mask_path, pred_mask)
            
            # Lưu mask màu
            cv2.imwrite(colored_path, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
            
            # Lưu mask tăng cường
            cv2.imwrite(enhanced_path, enhanced_mask)
            
            # Lưu overlay
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            
            # Lưu kết quả
            all_results[filename] = class_areas
            
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {filename}: {e}")
    
    # Lưu kết quả vào file CSV
    save_results_to_csv(all_results, output_dir)
    
    return all_results

def display_model_info(model):
    """Hiển thị thông tin về mô hình."""
    # In tóm tắt mô hình
    model.summary()
    
    # Lấy các lớp của mô hình
    print("\nCấu trúc mô hình:")
    for i, layer in enumerate(model.layers):
        print(f"{i+1}. {layer.name} - {layer.__class__.__name__} - Output shape: {layer.output_shape}")
    
    # Thông tin về tổng số tham số
    trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    non_trainable_params = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    
    print(f"\nTổng số tham số: {trainable_params + non_trainable_params:,}")
    print(f"Số tham số có thể huấn luyện: {trainable_params:,}")
    print(f"Số tham số không thể huấn luyện: {non_trainable_params:,}")

def main():
    # Thiết lập seed cho tính lặp lại
    SEED = 42
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Thử nghiệm mô hình phân đoạn bệnh trên da xoài')
    parser.add_argument('--model', type=str, default='models/unet_model.h5',
                        help='Đường dẫn đến mô hình đã huấn luyện')
    parser.add_argument('--config', type=str, default='configs/segmentation_config_new.yaml',
                        help='Đường dẫn đến file cấu hình')
    parser.add_argument('--mode', type=str, choices=['single', 'batch', 'evaluate', 'info'],
                        default='single', help='Chế độ thử nghiệm')
    parser.add_argument('--input', type=str, help='Đường dẫn đến ảnh hoặc thư mục ảnh đầu vào')
    parser.add_argument('--output', type=str, default='results',
                        help='Đường dẫn đến thư mục đầu ra')
    parser.add_argument('--test_dir', type=str, default='data/segmentation/test/images',
                        help='Đường dẫn đến thư mục ảnh test')
    parser.add_argument('--mask_dir', type=str, default='data/segmentation/test/masks',
                        help='Đường dẫn đến thư mục mask test')
    
    args = parser.parse_args()
    
    # Kiểm tra các file và thư mục cần thiết
    if not os.path.exists(args.model):
        print(f"Lỗi: Không tìm thấy mô hình tại {args.model}")
        return
    
    if not os.path.exists(args.config):
        print(f"Lỗi: Không tìm thấy file cấu hình tại {args.config}")
        return
    
    # Tải file cấu hình
    config = load_config(args.config)
    img_size = tuple(config['model']['input_shape'][:2])
    num_classes = config['model']['num_classes']
    
    print(f"Kích thước ảnh: {img_size}")
    print(f"Số lớp: {num_classes}")
    print(f"Tên các lớp: {config['model']['class_names']}")
    
    # Tải mô hình
    model = load_segmentation_model(args.model)
    
    # Xử lý theo chế độ
    if args.mode == 'info':
        # Hiển thị thông tin mô hình
        display_model_info(model)
    
    elif args.mode == 'single':
        # Kiểm tra ảnh đầu vào
        if not args.input or not os.path.exists(args.input):
            print(f"Lỗi: Không tìm thấy ảnh đầu vào tại {args.input}")
            return
        
        # Dự đoán trên một ảnh
        try:
            img, pred_mask, colored_mask, enhanced_mask, overlay, class_areas = predict_segmentation(
                model, args.input, img_size=img_size
            )
            
            # Tạo thư mục đầu ra
            if args.output:
                os.makedirs(args.output, exist_ok=True)
                os.makedirs(os.path.join(args.output, "masks"), exist_ok=True)
                os.makedirs(os.path.join(args.output, "colored"), exist_ok=True)
                os.makedirs(os.path.join(args.output, "enhanced"), exist_ok=True)
                os.makedirs(os.path.join(args.output, "overlays"), exist_ok=True)
                
                # Lưu kết quả
                filename = os.path.basename(args.input)
                cv2.imwrite(os.path.join(args.output, "masks", f"mask_{filename}"), pred_mask)
                cv2.imwrite(os.path.join(args.output, "colored", f"colored_{filename}"), 
                           cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(args.output, "enhanced", f"enhanced_{filename}"), enhanced_mask)
                cv2.imwrite(os.path.join(args.output, "overlays", f"overlay_{filename}"), 
                           cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            
            # Hiển thị kết quả
            display_results(img, pred_mask, colored_mask, enhanced_mask, overlay, class_areas, 
                           args.output, os.path.basename(args.input))
        
        except Exception as e:
            print(f"Lỗi khi dự đoán: {e}")
            import traceback
            traceback.print_exc()
    
    elif args.mode == 'batch':
        # Kiểm tra thư mục đầu vào
        if not args.input or not os.path.isdir(args.input):
            print(f"Lỗi: Không tìm thấy thư mục đầu vào tại {args.input}")
            return
        
        # Xử lý hàng loạt ảnh
        results = batch_process_images(model, args.input, args.output, img_size=img_size)
    
    elif args.mode == 'evaluate':
        # Kiểm tra thư mục test và mask
        if not os.path.isdir(args.test_dir):
            print(f"Lỗi: Không tìm thấy thư mục test tại {args.test_dir}")
            return
        
        if not os.path.isdir(args.mask_dir):
            print(f"Lỗi: Không tìm thấy thư mục mask tại {args.mask_dir}")
            return
        
        # Đánh giá mô hình
        metrics_per_class, avg_metrics = evaluate_on_test_set(
            model, args.test_dir, args.mask_dir, img_size=img_size, num_classes=num_classes
        )
        
        # Hiển thị kết quả
        print("\nKết quả đánh giá trung bình:")
        print(f"Mean IoU: {avg_metrics['mean_iou']:.4f}")
        print(f"Mean Dice: {avg_metrics['mean_dice']:.4f}")
        print(f"Pixel Accuracy: {avg_metrics['pixel_accuracy']:.4f}")
        
        print("\nKết quả đánh giá cho từng lớp:")
        print(f"{'Lớp':<15} {'IoU':<10} {'Dice':<10}")
        print("-" * 35)
        
        for class_name in CLASS_NAMES:
            iou = metrics_per_class['iou'][class_name]
            dice = metrics_per_class['dice'][class_name]
            print(f"{class_name:<15} {iou:.4f}{'':6} {dice:.4f}")
        
        # Lưu kết quả
        os.makedirs(args.output, exist_ok=True)
        result_path = os.path.join(args.output, "evaluation_results.txt")
        
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write("Kết quả đánh giá trung bình:\n")
            f.write(f"Mean IoU: {avg_metrics['mean_iou']:.4f}\n")
            f.write(f"Mean Dice: {avg_metrics['mean_dice']:.4f}\n")
            f.write(f"Pixel Accuracy: {avg_metrics['pixel_accuracy']:.4f}\n\n")
            
            f.write("Kết quả đánh giá cho từng lớp:\n")
            f.write(f"{'Lớp':<15} {'IoU':<10} {'Dice':<10}\n")
            f.write("-" * 35 + "\n")
            
            for class_name in CLASS_NAMES:
                iou = metrics_per_class['iou'][class_name]
                dice = metrics_per_class['dice'][class_name]
                f.write(f"{class_name:<15} {iou:.4f}{'':6} {dice:.4f}\n")
        
        print(f"\nĐã lưu kết quả đánh giá vào {result_path}")

if __name__ == "__main__":
    print("=" * 50)
    print("Thử nghiệm mô hình phân đoạn bệnh trên da xoài")
    print("=" * 50)
    main()