# File: src/models/transfer_learning.py
# Mô hình sử dụng transfer learning

import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization

def build_transfer_model(input_shape=(224, 224, 3), num_classes=4, base_model='mobilenetv2', model_config=None):
    """Xây dựng mô hình sử dụng transfer learning."""
    
    if model_config is None:
        model_config = {}
    
    # Lấy cấu hình hoặc sử dụng giá trị mặc định
    dense_units = model_config.get('dense_units', [1024, 512])
    dropout_rate = model_config.get('dropout_rate', 0.5)
    trainable_base = model_config.get('trainable_base', False)
    use_batch_norm = model_config.get('use_batch_norm', True)

    # Chọn mô hình cơ sở
    if base_model.lower() == 'vgg16':
        base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model.lower() == 'resnet50':
        base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model.lower() == 'efficientnet':
        base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    else:  # default to mobilenetv2
        base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Đóng băng các lớp của mô hình cơ sở
    base.trainable = trainable_base
    
    # Thêm các lớp mới
    x = base.output
    x = GlobalAveragePooling2D()(x)
    
    # Thêm các lớp Dense
    for units in dense_units:
        x = Dense(units, activation='relu')(x)
        if use_batch_norm:
            x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
    
    # Lớp đầu ra
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Tạo mô hình mới
    model = Model(inputs=base.input, outputs=predictions)
    
    return model