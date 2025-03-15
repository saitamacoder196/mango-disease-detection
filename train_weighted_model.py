# train_weighted_model.py
import os
import tensorflow as tf
import numpy as np
import yaml
import segmentation_models as sm
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from datetime import datetime
import cv2
import glob
from tensorflow.keras.utils import to_categorical

# Thiết lập seed cho tính khả tái
np.random.seed(42)
tf.random.set_seed(42)

# Đọc cấu hình
with open('configs/segmentation_config_new.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Các hàm metrics và loss tùy chỉnh
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

# Hàm loss có trọng số
def weighted_categorical_crossentropy(weights):
    weights = tf.convert_to_tensor(weights, dtype=tf.float32)
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        weighted_losses = y_true * tf.math.log(y_pred) * weights
        return -tf.reduce_sum(weighted_losses, axis=-1)
    return loss

# Tạo focal loss để tập trung vào các mẫu khó
def categorical_focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow(1 - y_pred, gamma)
        return tf.reduce_sum(weight * cross_entropy, axis=-1)
    return focal_loss

# Tạo thư mục lưu mô hình
model_dir = os.path.join(config['model']['save_dir'])
os.makedirs(model_dir, exist_ok=True)

# Hàm tạo bộ dữ liệu từ thư mục
def create_dataset(dir_path, batch_size=8, img_size=(512, 512), num_classes=6, is_training=False):
    # Lấy danh sách ảnh và mask
    img_dir = os.path.join(dir_path, 'images')
    mask_dir = os.path.join(dir_path, 'masks')
    
    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.*')))
    mask_paths = []
    
    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(mask_dir, mask_name)
        if os.path.exists(mask_path):
            mask_paths.append(mask_path)
        else:
            # Nếu không tìm thấy mask tương ứng, bỏ ảnh này
            img_paths.remove(img_path)
    
    print(f"Tìm thấy {len(img_paths)} cặp ảnh và mask trong thư mục {dir_path}")
    
    # Tạo dataset
    def data_generator():
        indices = np.arange(len(img_paths))
        if is_training:
            np.random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_size_actual = len(batch_indices)
            
            batch_imgs = np.zeros((batch_size_actual, *img_size, 3), dtype=np.float32)
            batch_masks = np.zeros((batch_size_actual, *img_size, num_classes), dtype=np.float32)
            
            for j, idx in enumerate(batch_indices):
                # Đọc ảnh
                img = cv2.imread(img_paths[idx])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                
                # Đọc mask
                mask = cv2.imread(mask_paths[idx], cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)
                
                # Áp dụng augmentation nếu là tập huấn luyện
                if is_training:
                    # Random horizontal flip
                    if np.random.random() > 0.5:
                        img = cv2.flip(img, 1)
                        mask = cv2.flip(mask, 1)
                    
                    # Random rotation
                    if np.random.random() > 0.5:
                        angle = np.random.uniform(-20, 20)
                        M = cv2.getRotationMatrix2D((img_size[0]//2, img_size[1]//2), angle, 1.0)
                        img = cv2.warpAffine(img, M, img_size)
                        mask = cv2.warpAffine(mask, M, img_size, flags=cv2.INTER_NEAREST)
                    
                    # Random brightness and contrast
                    if np.random.random() > 0.5:
                        alpha = np.random.uniform(0.8, 1.2)  # contrast
                        beta = np.random.uniform(-20, 20)    # brightness
                        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                
                # Chuẩn hóa ảnh
                img = img / 255.0
                
                # One-hot encoding mask
                mask_one_hot = np.zeros((*img_size, num_classes), dtype=np.float32)
                for c in range(num_classes):
                    mask_one_hot[:, :, c] = (mask == c).astype(np.float32)
                
                # Thêm vào batch
                batch_imgs[j] = img
                batch_masks[j] = mask_one_hot
            
            yield batch_imgs, batch_masks
    
    # Tính số bước mỗi epoch
    steps_per_epoch = (len(img_paths) + batch_size - 1) // batch_size
    
    # Tạo dataset từ generator
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=(
            (None, *img_size, 3),
            (None, *img_size, num_classes)
        )
    )
    
    if is_training:
        dataset = dataset.repeat()  # Lặp vô hạn cho tập huấn luyện
    
    return dataset, steps_per_epoch

# Tạo datasets
print("Tạo dataset huấn luyện và validation...")
train_dataset, train_steps = create_dataset(
    config['data']['train_dir'],
    batch_size=config['training']['batch_size'],
    img_size=tuple(config['data']['img_size']),
    num_classes=config['model']['num_classes'],
    is_training=True
)

val_dataset, val_steps = create_dataset(
    config['data']['validation_dir'],
    batch_size=config['training']['batch_size'],
    img_size=tuple(config['data']['img_size']),
    num_classes=config['model']['num_classes'],
    is_training=False
)

# Tạo mô hình
print("Xây dựng mô hình...")
model = sm.Unet(
    encoder_name=config['model']['segmentation_model']['encoder'],
    encoder_weights='imagenet',
    classes=config['model']['num_classes'],
    activation='softmax',
    input_shape=tuple(config['model']['input_shape'])
)

# Trọng số cho các lớp
class_weights = config['training'].get('class_weights', [0.01, 10.0, 20.0, 10.0, 15.0, 20.0])
print(f"Sử dụng trọng số lớp: {class_weights}")

# Biên dịch mô hình với weighted loss
loss_function = weighted_categorical_crossentropy(class_weights)
model.compile(
    optimizer=Adam(learning_rate=config['training']['learning_rate']),
    loss=loss_function,
    metrics=[iou_score, f1_score, 'accuracy']
)

# Các callback
callbacks = [
    ModelCheckpoint(
        os.path.join(model_dir, 'weighted_segmentation_model.h5'),
        monitor='val_iou_score',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_iou_score',
        mode='max',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    ),
    TensorBoard(
        log_dir=os.path.join('logs', 'segmentation', datetime.now().strftime("%Y%m%d-%H%M%S")),
        update_freq='epoch'
    )
]

# Huấn luyện mô hình
print("Bắt đầu huấn luyện...")
history = model.fit(
    train_dataset,
    steps_per_epoch=train_steps,
    validation_data=val_dataset,
    validation_steps=val_steps,
    epochs=150,  # Tăng epochs để mô hình có thời gian học
    callbacks=callbacks,
    verbose=1
)

# Vẽ đồ thị huấn luyện
plt.figure(figsize=(16, 6))

# Loss
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# IoU Score
plt.subplot(1, 3, 2)
plt.plot(history.history['iou_score'], label='Train IoU')
plt.plot(history.history['val_iou_score'], label='Validation IoU')
plt.title('IoU Score')
plt.xlabel('Epoch')
plt.ylabel('IoU Score')
plt.legend()

# F1 Score
plt.subplot(1, 3, 3)
plt.plot(history.history['f1_score'], label='Train F1')
plt.plot(history.history['val_f1_score'], label='Validation F1')
plt.title('F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(model_dir, 'training_history.png'))

print(f"Huấn luyện hoàn tất. Mô hình đã được lưu tại: {os.path.join(model_dir, 'weighted_segmentation_model.h5')}")
print(f"Đồ thị huấn luyện đã được lưu tại: {os.path.join(model_dir, 'training_history.png')}")