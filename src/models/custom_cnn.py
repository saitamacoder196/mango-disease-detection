# File: src/models/custom_cnn.py
# Mô hình CNN tự thiết kế

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

def build_custom_cnn(input_shape=(224, 224, 3), num_classes=4, model_config=None):
    """Xây dựng mô hình CNN tùy chỉnh."""
    
    if model_config is None:
        model_config = {}
    
    # Lấy cấu hình hoặc sử dụng giá trị mặc định
    filters = model_config.get('filters', [32, 64, 128, 256])
    dense_units = model_config.get('dense_units', [512, 256])
    dropout_rates = model_config.get('dropout_rates', [0.25, 0.25, 0.25, 0.5])
    use_batch_norm = model_config.get('use_batch_norm', True)
    
    model = Sequential()
    
    # Block 1
    model.add(Conv2D(filters[0], (3, 3), activation='relu', padding='same', input_shape=input_shape))
    if use_batch_norm:
        model.add(BatchNormalization())
    model.add(Conv2D(filters[0], (3, 3), activation='relu', padding='same'))
    if use_batch_norm:
        model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rates[0]))
    
    # Block 2
    model.add(Conv2D(filters[1], (3, 3), activation='relu', padding='same'))
    if use_batch_norm:
        model.add(BatchNormalization())
    model.add(Conv2D(filters[1], (3, 3), activation='relu', padding='same'))
    if use_batch_norm:
        model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rates[1]))
    
    # Block 3
    model.add(Conv2D(filters[2], (3, 3), activation='relu', padding='same'))
    if use_batch_norm:
        model.add(BatchNormalization())
    model.add(Conv2D(filters[2], (3, 3), activation='relu', padding='same'))
    if use_batch_norm:
        model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rates[2]))
    
    # Block 4 (optional)
    if len(filters) > 3:
        model.add(Conv2D(filters[3], (3, 3), activation='relu', padding='same'))
        if use_batch_norm:
            model.add(BatchNormalization())
        model.add(Conv2D(filters[3], (3, 3), activation='relu', padding='same'))
        if use_batch_norm:
            model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout_rates[3] if len(dropout_rates) > 3 else 0.5))
    
    # Fully connected layers
    model.add(Flatten())
    
    for i, units in enumerate(dense_units):
        model.add(Dense(units, activation='relu'))
        if use_batch_norm:
            model.add(BatchNormalization())
        if i < len(dense_units) - 1:  # Apply dropout to all but the last dense layer
            model.add(Dropout(dropout_rates[-1]))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    return model