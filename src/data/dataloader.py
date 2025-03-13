# File: src/data/dataloader.py
# Bộ nạp dữ liệu

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_data_generators(train_dir, validation_dir, test_dir=None, img_size=(224, 224), batch_size=32, augmentation=True):
    """Tạo data generators cho huấn luyện, xác thực và kiểm tra."""
    
    # Tạo ImageDataGenerator cho huấn luyện với tăng cường dữ liệu
    if augmentation:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)
    
    # Tạo ImageDataGenerator cho xác thực và kiểm tra (chỉ rescale)
    valid_datagen = ImageDataGenerator(rescale=1./255)
    
    # Tạo generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator = valid_datagen.flow_from_directory(
        validation_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Tạo test generator nếu test_dir được cung cấp
    test_generator = None
    if test_dir:
        test_generator = valid_datagen.flow_from_directory(
            test_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
    
    return train_generator, validation_generator, test_generator