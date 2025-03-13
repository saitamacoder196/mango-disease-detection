# File: segmentation_main.py
# Entry point cho phân đoạn bệnh xoài

import argparse
import os
import yaml
import tensorflow as tf
from src.data.segmentation_dataloader import create_segmentation_datasets
from src.models.segmentation_model import (
    build_segmentation_model, 
    get_segmentation_metrics, 
    get_segmentation_loss
)
from tensorflow.keras.callbacks import (
    ModelCheckpoint, 
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import matplotlib.pyplot as plt
import numpy as np
import cv2
from datetime import datetime
import segmentation_models as sm

def train_segmentation_model(config_path):
    """
    Huấn luyện mô hình phân đoạn.
    
    Args:
        config_path: Đường dẫn đến file cấu hình
    """
    # Đọc cấu hình
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Lấy cấu hình
    data_config = config['data']
    model_config = config['model']
    training_config = config['training']
    segmentation_config = model_config['segmentation_model']
    
    # Chuẩn bị dữ liệu
    train_dataset, val_dataset, _, train_steps, val_steps = create_segmentation_datasets(
        train_dir=data_config['train_dir'],
        validation_dir=data_config['validation_dir'],
        test_dir=data_config['test_dir'],
        img_size=tuple(data_config['img_size']),
        batch_size=training_config['batch_size'],
        num_classes=model_config['num_classes'],
        augmentation=training_config['use_augmentation'],
        augmentation_config=config['augmentation']
    )
    
    # Xây dựng mô hình
    model = build_segmentation_model(
        input_shape=tuple(model_config['input_shape']),
        num_classes=model_config['num_classes'],
        architecture=segmentation_config['architecture'],
        encoder=segmentation_config['encoder'],
        encoder_weights=segmentation_config['encoder_weights'],
        activation=segmentation_config['activation']
    )
    
    # Chọn optimizer
    if training_config['optimizer'].lower() == 'adam':
        optimizer = Adam(learning_rate=training_config['learning_rate'])
    elif training_config['optimizer'].lower() == 'rmsprop':
        optimizer = RMSprop(learning_rate=training_config['learning_rate'])
    elif training_config['optimizer'].lower() == 'sgd':
        optimizer = SGD(learning_rate=training_config['learning_rate'], momentum=0.9)
    else:
        optimizer = Adam(learning_rate=training_config['learning_rate'])
    
    # Biên dịch mô hình
    model.compile(
        optimizer=optimizer,
        loss=get_segmentation_loss(training_config['loss'], training_config['class_weights']),
        metrics=get_segmentation_metrics()
    )
    
    # In thông tin mô hình
    model.summary()
    
    # Tạo các callbacks
    callbacks = []
    
    # Model checkpoint
    os.makedirs(model_config['save_dir'], exist_ok=True)
    model_path = os.path.join(model_config['save_dir'], 'segmentation_model.h5')
    callbacks.append(ModelCheckpoint(
        model_path,
        monitor='val_iou_score',
        mode='max',
        save_best_only=True,
        verbose=1
    ))
    
    # Early stopping
    callbacks.append(EarlyStopping(
        monitor='val_loss',
        patience=training_config['early_stopping_patience'],
        restore_best_weights=True,
        verbose=1
    ))
    
    # Reduce LR on plateau
    callbacks.append(ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=training_config['reduce_lr_patience'],
        min_lr=1e-7,
        verbose=1
    ))
    
    # TensorBoard
    if training_config['use_tensorboard']:
        log_dir = os.path.join('logs', 'segmentation', datetime.now().strftime("%Y%m%d-%H%M%S"))
        callbacks.append(TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            update_freq='epoch'
        ))
    
    # Huấn luyện mô hình
    history = model.fit(
        train_dataset,
        steps_per_epoch=train_steps,
        validation_data=val_dataset,
        validation_steps=val_steps,
        epochs=training_config['epochs'],
        callbacks=callbacks
    )
    
    # Lưu lịch sử huấn luyện
    save_training_plots(history, model_config['save_dir'])
    
    print(f"Đã huấn luyện xong mô hình và lưu tại {model_path}")
    
    return model, history

def save_training_plots(history, save_dir):
    """Lưu biểu đồ quá trình huấn luyện."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Tạo biểu đồ loss
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Tạo biểu đồ IoU Score
    plt.subplot(1, 2, 2)
    plt.plot(history.history['iou_score'], label='Train IoU')
    plt.plot(history.history['val_iou_score'], label='Validation IoU')
    plt.title('IoU Score')
    plt.xlabel('Epoch')
    plt.ylabel('IoU Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'segmentation_training_history.png'))
    plt.close()

def evaluate_segmentation_model(model_path, config_path):
    """
    Đánh giá mô hình phân đoạn.
    
    Args:
        model_path: Đường dẫn đến mô hình đã huấn luyện
        config_path: Đường dẫn đến file cấu hình
    """
    # Đọc cấu hình
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Lấy cấu hình
    data_config = config['data']
    model_config = config['model']
    evaluation_config = config['evaluation']
    
    # Chuẩn bị dữ liệu kiểm thử
    _, _, test_dataset, _, test_steps = create_segmentation_datasets(
        train_dir=None,
        validation_dir=None,
        test_dir=data_config['test_dir'],
        img_size=tuple(data_config['img_size']),
        batch_size=evaluation_config['batch_size'],
        num_classes=model_config['num_classes'],
        augmentation=False
    )
    
    # Tải mô hình
    print(f"Đang tải mô hình từ {model_path}...")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'iou_score': sm.metrics.IOUScore(threshold=0.5),
            'f1-score': sm.metrics.FScore(threshold=0.5)
        }
    )
    
    # Đánh giá mô hình
    print("Đang đánh giá mô hình...")
    results = model.evaluate(test_dataset, steps=test_steps)
    
    # In kết quả
    metrics_names = model.metrics_names
    for name, value in zip(metrics_names, results):
        print(f"{name}: {value:.4f}")
    
    # Lưu kết quả
    os.makedirs(evaluation_config['save_dir'], exist_ok=True)
    with open(os.path.join(evaluation_config['save_dir'], 'evaluation_results.txt'), 'w') as f:
        for name, value in zip(metrics_names, results):
            f.write(f"{name}: {value:.4f}\n")
    
    # Hiển thị một số ví dụ
    visualize_predictions(model, data_config['test_dir'], model_config, evaluation_config['save_dir'])
    
    return results

def visualize_predictions(model, test_dir, model_config, save_dir):
    """Hiển thị kết quả dự đoán trên một số mẫu."""
    # Đường dẫn thư mục
    images_dir = os.path.join(test_dir, 'images')
    masks_dir = os.path.join(test_dir, 'masks')
    
    # Lấy danh sách file ảnh
    image_files = sorted([f for f in os.listdir(images_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    # Chọn ngẫu nhiên một số mẫu
    num_samples = min(5, len(image_files))
    sample_indices = np.random.choice(len(image_files), num_samples, replace=False)
    
    # Lấy tên lớp
    class_names = model_config['class_names']
    
    # Màu cho các lớp (RGB)
    colors = [
        [0, 0, 0],      # Background - đen
        [255, 0, 0],    # Da cám - đỏ
        [0, 255, 0],    # Da ếch - xanh lá
        [0, 0, 255],    # Đóm đen - xanh dương
        [255, 255, 0],  # Thán thư - vàng
        [255, 0, 255]   # Rùi đụt - tím
    ]
    
    plt.figure(figsize=(15, 4 * num_samples))
    
    for i, idx in enumerate(sample_indices):
        image_file = image_files[idx]
        mask_file = os.path.splitext(image_file)[0] + '.png'
        
        # Đọc ảnh và mask
        img_path = os.path.join(images_dir, image_file)
        mask_path = os.path.join(masks_dir, mask_file)
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, tuple(model_config['input_shape'][:2]))
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, tuple(model_config['input_shape'][:2]), interpolation=cv2.INTER_NEAREST)
        
        # Dự đoán
        img_input = img / 255.0
        img_input = np.expand_dims(img_input, axis=0)
        
        pred_mask = model.predict(img_input)[0]
        pred_mask = np.argmax(pred_mask, axis=-1)
        
        # Tạo hình ảnh mask màu
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        colored_pred = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        
        for class_idx, color in enumerate(colors):
            colored_mask[mask == class_idx] = color
            colored_pred[pred_mask == class_idx] = color
        
        # Tạo overlay
        alpha = 0.6
        overlay = cv2.addWeighted(img, 1-alpha, colored_pred, alpha, 0)
        
        # Hiển thị
        plt.subplot(num_samples, 3, 3*i+1)
        plt.imshow(img)
        plt.title(f"Ảnh gốc - {image_file}")
        plt.axis('off')
        
        plt.subplot(num_samples, 3, 3*i+2)
        plt.imshow(colored_mask)
        plt.title("Mask thực tế")
        plt.axis('off')
        
        plt.subplot(num_samples, 3, 3*i+3)
        plt.imshow(overlay)
        plt.title("Dự đoán")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'segmentation_examples.png'))
    plt.close()

def predict_segmentation(image_path, model_path, config_path, output_path=None, overlay=True):
    """
    Dự đoán phân đoạn cho một ảnh.
    
    Args:
        image_path: Đường dẫn đến ảnh cần dự đoán
        model_path: Đường dẫn đến mô hình đã huấn luyện
        config_path: Đường dẫn đến file cấu hình
        output_path: Đường dẫn lưu kết quả (tùy chọn)
        overlay: Có tạo overlay hay không
        
    Returns:
        pred_mask: Mask dự đoán
        overlay_img: Ảnh overlay nếu overlay=True
    """
    # Đọc cấu hình
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Lấy cấu hình
    model_config = config['model']
    
    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh từ {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize ảnh
    img_size = tuple(model_config['input_shape'][:2])
    img_resized = cv2.resize(img, img_size)
    
    # Chuẩn bị đầu vào
    img_input = img_resized / 255.0
    img_input = np.expand_dims(img_input, axis=0)
    
    # Tải mô hình
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'iou_score': sm.metrics.IOUScore(threshold=0.5),
            'f1-score': sm.metrics.FScore(threshold=0.5)
        }
    )
    
    # Dự đoán
    pred = model.predict(img_input)[0]
    pred_mask = np.argmax(pred, axis=-1)
    
    # Màu cho các lớp (BGR cho OpenCV)
    colors = [
        [0, 0, 0],      # Background - đen
        [0, 0, 255],    # Da cám - đỏ
        [0, 255, 0],    # Da ếch - xanh lá
        [255, 0, 0],    # Đóm đen - xanh dương
        [0, 255, 255],  # Thán thư - vàng
        [255, 0, 255]   # Rùi đụt - tím
    ]
    
    # Tạo mask màu
    colored_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    for class_idx, color in enumerate(colors):
        colored_mask[pred_mask == class_idx] = color
    
    # Tạo overlay
    overlay_img = None
    if overlay:
        alpha = 0.6
        overlay_img = cv2.addWeighted(img_resized, 1-alpha, colored_mask, alpha, 0)
        overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
    
    # Lưu kết quả nếu cần
    if output_path:
        if overlay:
            cv2.imwrite(output_path, overlay_img)
        else:
            cv2.imwrite(output_path, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
    
    return pred_mask, overlay_img if overlay else colored_mask

def main():
    parser = argparse.ArgumentParser(description='Phân đoạn bệnh trên xoài')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'evaluate', 'predict'],
                        help='Chế độ: train, evaluate, predict')
    parser.add_argument('--config', type=str, default='configs/segmentation_config.yaml',
                        help='Đường dẫn đến file cấu hình')
    parser.add_argument('--model_path', type=str,
                        help='Đường dẫn đến mô hình đã huấn luyện (cho evaluate và predict)')
    parser.add_argument('--image_path', type=str,
                        help='Đường dẫn đến ảnh cần dự đoán (chỉ cho predict)')
    parser.add_argument('--output_path', type=str,
                        help='Đường dẫn lưu kết quả dự đoán (chỉ cho predict)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_segmentation_model(args.config)
    
    elif args.mode == 'evaluate':
        if not args.model_path:
            parser.error("--model_path là bắt buộc trong chế độ evaluate")
        evaluate_segmentation_model(args.model_path, args.config)
    
    elif args.mode == 'predict':
        if not args.model_path or not args.image_path:
            parser.error("--model_path và --image_path là bắt buộc trong chế độ predict")
        
        output_path = args.output_path or 'segmentation_result.png'
        _, overlay = predict_segmentation(args.image_path, args.model_path, args.config, output_path)
        
        print(f"Đã lưu kết quả dự đoán tại {output_path}")
        
        # Hiển thị kết quả nếu chạy trên môi trường có giao diện
        try:
            plt.figure(figsize=(10, 5))
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title("Kết quả phân đoạn")
            plt.show()
        except:
            pass

if __name__ == "__main__":
    main()