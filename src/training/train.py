# File: src/training/train.py
# Huấn luyện mô hình

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
from src.data.dataloader import create_data_generators

def train_model(model, train_dir, validation_dir, batch_size=32, epochs=50, learning_rate=0.001, 
               model_save_path='models/model.h5', training_config=None):
    """Huấn luyện mô hình."""
    
    if training_config is None:
        training_config = {}
    
    # Lấy cấu hình hoặc sử dụng giá trị mặc định
    img_size = training_config.get('img_size', (224, 224))
    use_augmentation = training_config.get('use_augmentation', True)
    early_stopping_patience = training_config.get('early_stopping_patience', 10)
    reduce_lr_patience = training_config.get('reduce_lr_patience', 5)
    use_tensorboard = training_config.get('use_tensorboard', True)
    class_weights = training_config.get('class_weights', None)
    
    # Tạo data generators
    train_generator, validation_generator, _ = create_data_generators(
        train_dir=train_dir,
        validation_dir=validation_dir,
        test_dir=None,
        img_size=img_size,
        batch_size=batch_size,
        augmentation=use_augmentation
    )
    
    # Biên dịch mô hình
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    # Đảm bảo thư mục lưu trữ tồn tại
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Tạo callbacks
    callbacks = [
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Thêm TensorBoard callback nếu được yêu cầu
    if use_tensorboard:
        log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard_callback)
    
    # Huấn luyện mô hình
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    # Lưu biểu đồ huấn luyện
    save_training_plots(history, os.path.dirname(model_save_path))
    
    return history

def save_training_plots(history, save_dir):
    """Lưu biểu đồ quá trình huấn luyện."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Tạo biểu đồ accuracy
    plt.figure(figsize=(12, 5))
    
    # Biểu đồ accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Biểu đồ loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Lưu biểu đồ
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()
    
    # Tạo biểu đồ Precision và Recall
    if 'precision' in history.history and 'recall' in history.history:
        plt.figure(figsize=(12, 5))
        
        # Biểu đồ precision
        plt.subplot(1, 2, 1)
        plt.plot(history.history['precision'], label='Train Precision')
        plt.plot(history.history['val_precision'], label='Validation Precision')
        plt.title('Model Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend()
        
        # Biểu đồ recall
        plt.subplot(1, 2, 2)
        plt.plot(history.history['recall'], label='Train Recall')
        plt.plot(history.history['val_recall'], label='Validation Recall')
        plt.title('Model Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend()
        
        # Lưu biểu đồ
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'precision_recall_history.png'))
        plt.close()