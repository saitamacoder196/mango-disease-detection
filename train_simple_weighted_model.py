#!/usr/bin/env python
# File: train_simple_weighted_model.py
# Script huấn luyện mô hình phân đoạn với trọng số cho lớp bệnh

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import glob
import yaml
import albumentations as A
import segmentation_models as sm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.utils import to_categorical
from datetime import datetime
from tqdm import tqdm
import sys

# Thiết lập seed cho tính lặp lại
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Kiểm tra phiên bản TensorFlow và GPU
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Hàm đọc file cấu hình
def load_config(config_path):
    """Đọc file cấu hình."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# DataGenerator cho bài toán segmentation
class SegmentationDataGenerator(tf.keras.utils.Sequence):
    """Bộ nạp dữ liệu cho mô hình phân đoạn."""
    
    def __init__(self, images_dir, masks_dir, batch_size=8, img_size=(512, 512), 
                num_classes=6, augmentation=False, augmentation_config=None, shuffle=True):
        """Khởi tạo generator."""
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.augmentation = augmentation
        
        # Lấy danh sách file ảnh
        self.image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")) + 
                                 glob.glob(os.path.join(images_dir, "*.jpeg")) + 
                                 glob.glob(os.path.join(images_dir, "*.png")))
        
        # Lấy danh sách file mask tương ứng
        self.mask_paths = []
        valid_image_paths = []
        
        for img_path in self.image_paths:
            img_name = os.path.basename(img_path)
            base_name = os.path.splitext(img_name)[0]
            mask_path = os.path.join(masks_dir, f"{base_name}.png")
            if os.path.exists(mask_path):
                self.mask_paths.append(mask_path)
                valid_image_paths.append(img_path)
        
        # Cập nhật lại danh sách ảnh hợp lệ
        self.image_paths = valid_image_paths
        
        # Tạo albumentation cho augmentation
        if augmentation:
            self.aug_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(p=0.5, shift_limit=0.1, scale_limit=0.2, rotate_limit=20),
                A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2),
                A.GaussianBlur(p=0.3, blur_limit=7),
                A.GridDistortion(p=0.3)
            ])
        
        # Tạo indices
        self.indices = np.arange(len(self.image_paths))
        self.on_epoch_end()
    
    def __len__(self):
        """Trả về số batch trong một epoch."""
        return max(1, len(self.image_paths) // self.batch_size)
    
    def __getitem__(self, index):
        """Trả về một batch dữ liệu."""
        # Lấy indices của batch hiện tại
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.indices))
        batch_indices = self.indices[start_idx:end_idx]
        
        # Số lượng ảnh thực tế trong batch
        actual_batch_size = len(batch_indices)
        
        # Khởi tạo batch data
        batch_imgs = np.zeros((actual_batch_size, *self.img_size, 3), dtype=np.float32)
        batch_masks = np.zeros((actual_batch_size, *self.img_size, self.num_classes), dtype=np.float32)
        
        # Nạp dữ liệu
        for i, idx in enumerate(batch_indices):
            # Đọc ảnh và mask
            img = cv2.imread(self.image_paths[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
            
            # Áp dụng augmentation nếu được yêu cầu
            if self.augmentation:
                augmented = self.aug_transform(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
            
            # Chuẩn hóa ảnh
            img = img / 255.0
            
            # One-hot encoding cho mask
            mask_onehot = to_categorical(mask, num_classes=self.num_classes)
            
            # Thêm vào batch
            batch_imgs[i] = img
            batch_masks[i] = mask_onehot
        
        return batch_imgs, batch_masks
    
    def on_epoch_end(self):
        """Được gọi khi kết thúc một epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)

# Tính trọng số cho các lớp
def calculate_class_weights(mask_paths, num_classes):
    """Tính trọng số cho các lớp dựa trên tỷ lệ xuất hiện."""
    class_counts = np.zeros(num_classes)
    
    print("Đang tính trọng số cho các lớp...")
    
    for mask_path in tqdm(mask_paths):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Không thể đọc mask {mask_path}")
            continue
        
        for class_idx in range(num_classes):
            class_counts[class_idx] += np.sum(mask == class_idx)
    
    # In số lượng pixel cho từng lớp
    print("Số lượng pixel cho từng lớp:")
    for class_idx in range(num_classes):
        print(f"Lớp {class_idx}: {class_counts[class_idx]} pixels")
    
    # Tính trọng số: nghịch đảo của tần suất
    total_pixels = np.sum(class_counts)
    class_frequencies = class_counts / total_pixels
    
    # Tránh chia cho 0
    class_frequencies = np.where(class_frequencies == 0, 1e-6, class_frequencies)
    
    # Trọng số dùng median frequency balancing
    median_freq = np.median(class_frequencies)
    class_weights = median_freq / class_frequencies
    
    # Điều chỉnh trọng số cho lớp background (class_idx = 0)
    # Giảm trọng số cho background, tăng trọng số cho vùng bệnh
    class_weights[0] = 1.0  # Giá trị tiêu chuẩn cho background
    for class_idx in range(1, num_classes):
        class_weights[class_idx] *= 5.0  # Tăng trọng số cho vùng bệnh lên 5 lần
    
    # Chuẩn hóa trọng số để tổng = num_classes
    class_weights = class_weights * num_classes / np.sum(class_weights)
    
    print("Trọng số cho các lớp:")
    for class_idx in range(num_classes):
        print(f"Lớp {class_idx}: {class_weights[class_idx]:.4f}")
    
    return class_weights

# Định nghĩa các loss function
def weighted_categorical_crossentropy(class_weights):
    """Hàm loss entropy chéo có trọng số cho các lớp."""
    def loss(y_true, y_pred):
        # Tạo tensor trọng số
        weights_tensor = tf.convert_to_tensor(class_weights, dtype=tf.float32)
        
        # reshape y_true thành [batch*height*width, num_classes]
        y_true = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
        y_pred = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
        
        # Nhân các trọng số với từng lớp
        weights = tf.reduce_sum(y_true * weights_tensor, axis=-1)
        
        # Tính loss cho từng pixel
        loss_per_pixel = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # Nhân loss với trọng số và tính trung bình
        weighted_loss = tf.reduce_mean(loss_per_pixel * weights)
        
        return weighted_loss
    
    return loss

def weighted_dice_loss(class_weights):
    """Hàm loss dice có trọng số cho các lớp."""
    def loss(y_true, y_pred, smooth=1e-6):
        # Tạo tensor trọng số
        weights_tensor = tf.convert_to_tensor(class_weights, dtype=tf.float32)
        
        # Chuyển đổi kích thước
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Tính Dice cho từng lớp
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        denominator = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
        
        # Thêm smooth để tránh chia cho 0
        dice_coef = (numerator + smooth) / (denominator + smooth)
        
        # Áp dụng trọng số cho từng lớp
        weighted_dice = dice_coef * weights_tensor
        
        # Tính trung bình
        dice_loss = 1 - tf.reduce_mean(weighted_dice)
        
        return dice_loss
    
    return loss

def combined_weighted_loss(class_weights):
    """Kết hợp weighted categorical crossentropy và weighted dice loss."""
    wce_loss = weighted_categorical_crossentropy(class_weights)
    wdice_loss = weighted_dice_loss(class_weights)
    
    def loss(y_true, y_pred):
        return wce_loss(y_true, y_pred) + wdice_loss(y_true, y_pred)
    
    return loss

# Lớp callback tùy chỉnh để vẽ biểu đồ trong quá trình huấn luyện
class SegmentationVisualCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, img_idx=0, save_dir='logs/visualization'):
        super().__init__()
        self.validation_data = validation_data
        self.img_idx = img_idx  # Chỉ số của ảnh trong validation_data để hiển thị
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
    
    def on_epoch_end(self, epoch, logs=None):
        # Lấy một batch dữ liệu validation
        x_val, y_val = self.validation_data[self.img_idx]
        
        # Kiểm tra kích thước batch
        if len(x_val) == 0:
            return
            
        # Chọn một ảnh từ batch để hiển thị
        img = x_val[0]
        true_mask = np.argmax(y_val[0], axis=-1)
        
        # Dự đoán mask
        pred_mask = self.model.predict(np.expand_dims(img, axis=0))[0]
        pred_mask = np.argmax(pred_mask, axis=-1)
        
        # Vẽ biểu đồ
        plt.figure(figsize=(12, 4))
        
        # Ảnh gốc
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title('Ảnh gốc')
        plt.axis('off')
        
        # Mask thực tế
        plt.subplot(1, 3, 2)
        plt.imshow(true_mask)
        plt.title('Mask thực tế')
        plt.axis('off')
        
        # Mask dự đoán
        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask)
        plt.title(f'Mask dự đoán (Epoch {epoch+1})')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'visualization_epoch_{epoch+1}.png'))
        plt.close()

class TrainingMonitor(tf.keras.callbacks.Callback):
    def __init__(self, figsize=(12, 8), save_dir='logs/training_monitor'):
        super().__init__()
        self.figsize = figsize
        self.save_dir = save_dir
        self.losses = []
        self.val_losses = []
        self.accuracies = []
        self.val_accuracies = []
        self.iou_scores = []
        self.val_iou_scores = []
        self.learning_rates = []
        
        os.makedirs(self.save_dir, exist_ok=True)
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Lưu các metric
        self.losses.append(logs.get('loss', 0))
        self.val_losses.append(logs.get('val_loss', 0))
        self.accuracies.append(logs.get('accuracy', 0))
        self.val_accuracies.append(logs.get('val_accuracy', 0))
        self.iou_scores.append(logs.get('iou_score', 0))
        self.val_iou_scores.append(logs.get('val_iou_score', 0))
        
        # Lấy learning rate hiện tại
        if hasattr(self.model.optimizer, 'lr'):
            lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
            self.learning_rates.append(lr)
        
        # Vẽ biểu đồ
        plt.figure(figsize=self.figsize)
        
        # Loss
        plt.subplot(2, 2, 1)
        plt.plot(self.losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Accuracy
        plt.subplot(2, 2, 2)
        plt.plot(self.accuracies, label='Train Accuracy')
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # IoU Score
        plt.subplot(2, 2, 3)
        plt.plot(self.iou_scores, label='Train IoU')
        plt.plot(self.val_iou_scores, label='Validation IoU')
        plt.title('IoU Score')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Learning Rate
        if self.learning_rates:
            plt.subplot(2, 2, 4)
            plt.plot(self.learning_rates)
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'training_monitor_epoch_{epoch+1}.png'))
        plt.close()

# Create a simpler UNet model using Keras directly to avoid the name conflict issue
def create_unet_model(input_shape, num_classes):
    from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, BatchNormalization
    from tensorflow.keras.models import Model
    
    inputs = Input(input_shape)
    
    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(0.25)(p1)
    
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(0.25)(p2)
    
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(0.25)(p3)
    
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(0.25)(p4)
    
    # Bridge
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    c5 = Dropout(0.5)(c5)
    
    # Decoder
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    c6 = Dropout(0.25)(c6)
    
    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    c7 = Dropout(0.25)(c7)
    
    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    c8 = Dropout(0.25)(c8)
    
    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c9)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Hàm chính để huấn luyện mô hình
def train_segmentation_model(config_path):
    """Huấn luyện mô hình phân đoạn với các cải tiến."""
    # Tải cấu hình
    config = load_config(config_path)
    
    # Lấy thông tin cấu hình
    data_config = config['data']
    model_config = config['model']
    training_config = config['training']
    augmentation_config = config.get('augmentation', {})
    
    # Đường dẫn dữ liệu
    train_dir = data_config['train_dir']
    val_dir = data_config['validation_dir']
    test_dir = data_config.get('test_dir')
    
    # Kích thước ảnh và số lớp
    img_size = tuple(data_config['img_size'])
    num_classes = model_config['num_classes']
    class_names = model_config['class_names']
    
    # Cấu hình huấn luyện
    batch_size = training_config['batch_size']
    epochs = training_config['epochs']
    learning_rate = training_config['learning_rate']
    use_augmentation = training_config.get('use_augmentation', True)
    
    # Đường dẫn đầy đủ cho train và validation
    train_images_dir = os.path.join(train_dir, 'images')
    train_masks_dir = os.path.join(train_dir, 'masks')
    val_images_dir = os.path.join(val_dir, 'images')
    val_masks_dir = os.path.join(val_dir, 'masks')
    
    # Kiểm tra thư mục dữ liệu
    for dir_path in [train_images_dir, train_masks_dir, val_images_dir, val_masks_dir]:
        if not os.path.exists(dir_path):
            print(f"Lỗi: Không tìm thấy thư mục {dir_path}")
            sys.exit(1)
    
    # Tạo data generator
    train_generator = SegmentationDataGenerator(
        train_images_dir, 
        train_masks_dir, 
        batch_size=batch_size, 
        img_size=img_size,
        num_classes=num_classes, 
        augmentation=use_augmentation,
        augmentation_config=augmentation_config
    )
    
    val_generator = SegmentationDataGenerator(
        val_images_dir, 
        val_masks_dir, 
        batch_size=batch_size, 
        img_size=img_size,
        num_classes=num_classes, 
        augmentation=False
    )
    
    # Tính trọng số cho các lớp
    # Lấy tất cả đường dẫn mask trong tập train
    train_mask_paths = [os.path.join(train_masks_dir, f) for f in os.listdir(train_masks_dir) 
                        if f.endswith('.png')]
    
    class_weights = calculate_class_weights(train_mask_paths, num_classes)
    
    # Tạo mô hình
    # Use Keras directly to avoid the name conflict error
    model = create_unet_model(
        input_shape=tuple(model_config['input_shape']),
        num_classes=num_classes
    )
    
    # Biên dịch mô hình với loss và metrics tùy chỉnh
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=combined_weighted_loss(class_weights),
        metrics=[
            'accuracy',
            create_custom_iou_metric(threshold=0.5),
            create_custom_f1_metric(threshold=0.5)
        ]
    )

    
    # Tóm tắt mô hình
    model.summary()
    
    # Tạo thư mục lưu mô hình
    model_save_dir = model_config['save_dir']
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Tạo callback
    callbacks = []
    
    # ModelCheckpoint - lưu mô hình tốt nhất
    checkpoint_path = os.path.join(model_save_dir, 'weighted_segmentation_model.h5')
    callbacks.append(ModelCheckpoint(
        checkpoint_path,
        monitor='val_iou_score',
        mode='max',
        save_best_only=True,
        verbose=1
    ))
    
    # Early stopping - ngừng khi không cải thiện
    callbacks.append(EarlyStopping(
        monitor='val_iou_score',
        mode='max',
        patience=training_config.get('early_stopping_patience', 20),
        restore_best_weights=True,
        verbose=1
    ))
    
    # Reduce LR on plateau - giảm learning rate khi không cải thiện
    callbacks.append(ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=training_config.get('reduce_lr_patience', 5),
        min_lr=1e-7,
        verbose=1
    ))
    
    # TensorBoard - để theo dõi quá trình huấn luyện
    if training_config.get('use_tensorboard', True):
        log_dir = os.path.join('logs', 'segmentation', datetime.now().strftime("%Y%m%d-%H%M%S"))
        callbacks.append(TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ))
    
    # TrainingMonitor - hiển thị biểu đồ trong quá trình huấn luyện
    callbacks.append(TrainingMonitor())
    
    # SegmentationVisualCallback - hiển thị kết quả phân đoạn trong quá trình huấn luyện
    callbacks.append(SegmentationVisualCallback(val_generator))
    
    # Huấn luyện mô hình
    print(f"Bắt đầu huấn luyện mô hình UNet-Keras")
    print(f"Số lượng ảnh train: {len(train_generator) * batch_size}")
    print(f"Số lượng ảnh validation: {len(val_generator) * batch_size}")
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Lưu biểu đồ huấn luyện
    plt.figure(figsize=(15, 5))
    
    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # IoU
    plt.subplot(1, 3, 2)
    plt.plot(history.history['iou_score'], label='Train IoU')
    plt.plot(history.history['val_iou_score'], label='Validation IoU')
    plt.title('IoU Score')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # F1-Score
    plt.subplot(1, 3, 3)
    plt.plot(history.history['f1-score'], label='Train F1')
    plt.plot(history.history['val_f1-score'], label='Validation F1')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_dir, 'weighted_training_history.png'))
    
    print(f"Huấn luyện hoàn tất! Mô hình đã được lưu tại {checkpoint_path}")
    
    return model, history
# Biên dịch mô hình với loss và metrics tùy chỉnh
def create_custom_iou_metric(threshold=0.5):
    def iou_score(y_true, y_pred):
        # Flatten the tensors
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        
        # Convert predictions to binary based on threshold
        y_pred_f = tf.cast(y_pred_f > threshold, tf.float32)
        
        # Calculate intersection and union
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
        
        # Calculate IoU score
        iou = (intersection + 1e-7) / (union + 1e-7)
        return iou
    
    return iou_score

def create_custom_f1_metric(threshold=0.5):
    def f1_score(y_true, y_pred):
        # Flatten the tensors
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        
        # Convert predictions to binary based on threshold
        y_pred_f = tf.cast(y_pred_f > threshold, tf.float32)
        
        # Calculate true positives, false positives, false negatives
        true_positives = tf.reduce_sum(y_true_f * y_pred_f)
        false_positives = tf.reduce_sum(y_pred_f) - true_positives
        false_negatives = tf.reduce_sum(y_true_f) - true_positives
        
        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives + 1e-7)
        recall = true_positives / (true_positives + false_negatives + 1e-7)
        
        # Calculate F1 score
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        return f1
    
    return f1_score

if __name__ == "__main__":
    config_path = "configs/segmentation_config_new.yaml"
    
    # Kiểm tra xem có tham số dòng lệnh không
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    
    # Kiểm tra file cấu hình tồn tại
    # Kiểm tra file cấu hình tồn tại
    if not os.path.exists(config_path):
        print(f"Lỗi: Không tìm thấy file cấu hình {config_path}")
        sys.exit(1)
    
    print(f"Sử dụng file cấu hình: {config_path}")
    
    # Huấn luyện mô hình
    train_segmentation_model(config_path)