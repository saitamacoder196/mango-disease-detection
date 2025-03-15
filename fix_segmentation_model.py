# fix_segmentation_model.py
import os
import tensorflow as tf
import segmentation_models as sm
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from src.data.segmentation_dataloader import create_segmentation_datasets

# Thiết lập segmentation-models
sm.set_framework('tf.keras')
sm.framework()

# Tải cấu hình
import yaml
with open('configs/segmentation_config_new.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Thiết lập trọng số lớp để tập trung hơn vào các lớp bệnh
class_weights = [0.1, 2.0, 2.0, 2.0, 2.0, 2.0]  # Điều chỉnh dựa trên phân phối lớp của bạn

# Tạo dataset với nhiều augmentation hơn
train_dataset, val_dataset, _, train_steps, val_steps = create_segmentation_datasets(
    train_dir=config['data']['train_dir'],
    validation_dir=config['data']['validation_dir'],
    test_dir=config['data']['test_dir'],
    img_size=tuple(config['data']['img_size']),
    batch_size=config['training']['batch_size'],
    num_classes=config['model']['num_classes'],
    augmentation=True,
    augmentation_config=config['augmentation']
)

# Tạo một mô hình mới
input_shape = tuple(config['model']['input_shape'])
num_classes = config['model']['num_classes']
architecture = config['model']['segmentation_model']['architecture']
encoder = config['model']['segmentation_model']['encoder']

model = sm.Unet(
    encoder_name=encoder,
    encoder_weights='imagenet',
    classes=num_classes,
    activation='softmax',
    input_shape=input_shape
)

# Biên dịch với hàm weighted categorical crossentropy
def weighted_categorical_crossentropy(weights):
    weights = tf.keras.backend.variable(weights)
    def loss(y_true, y_pred):
        y_pred /= tf.keras.backend.sum(y_pred, axis=-1, keepdims=True)
        y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        loss = y_true * tf.keras.backend.log(y_pred) * weights
        loss = -tf.keras.backend.sum(loss, -1)
        return loss
    return loss

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate']),
    loss=weighted_categorical_crossentropy(class_weights),
    metrics=[
        sm.metrics.IOUScore(threshold=0.5),
        sm.metrics.FScore(threshold=0.5),
        'accuracy'
    ]
)

# Tạo callbacks
callbacks = [
    ModelCheckpoint(
        'models/fixed_segmentation_model.h5',
        monitor='val_iou_score',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=15,  # Tăng patience
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# Huấn luyện mô hình
history = model.fit(
    train_dataset,
    steps_per_epoch=train_steps,
    validation_data=val_dataset,
    validation_steps=val_steps,
    epochs=100,  # Tăng số epochs
    callbacks=callbacks
)

print("Huấn luyện mô hình hoàn tất. Đã lưu vào models/fixed_segmentation_model.h5")