# File: src/inference/predict.py
# Dự đoán trên ảnh mới

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf

def preprocess_image_for_prediction(image_path, img_size=(224, 224)):
    """Tiền xử lý ảnh cho dự đoán."""
    # Đọc ảnh
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize ảnh
    img = cv2.resize(img, img_size)
    
    # Chuẩn hóa ảnh
    img = img / 255.0
    
    # Mở rộng kích thước batch
    img = np.expand_dims(img, axis=0)
    
    return img

def predict_image(model_path, image_path, img_size=(224, 224), class_names=None, visualize=True, save_path=None):
    """Dự đoán bệnh trên một ảnh mới."""
    # Tải mô hình
    model = load_model(model_path)
    
    # Tiền xử lý ảnh
    processed_img = preprocess_image_for_prediction(image_path, img_size)
    
    # Dự đoán
    prediction = model.predict(processed_img)[0]
    predicted_class_idx = np.argmax(prediction)
    confidence = prediction[predicted_class_idx]
    
    # Nếu không cung cấp tên lớp, sử dụng index
    if class_names is None:
        predicted_class = f"Class {predicted_class_idx}"
    else:
        predicted_class = class_names[predicted_class_idx]
    
    # Hiển thị kết quả
    result = {
        'class': predicted_class,
        'confidence': float(confidence),
        'probabilities': {
            class_names[i] if class_names else f"Class {i}": float(prob)
            for i, prob in enumerate(prediction)
        }
    }
    
    # Hiển thị ảnh và dự đoán
    if visualize:
        # Đọc ảnh gốc
        original_img = cv2.imread(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 6))
        
        # Hiển thị ảnh
        plt.subplot(1, 2, 1)
        plt.imshow(original_img)
        plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}")
        plt.axis('off')
        
        # Hiển thị biểu đồ xác suất
        plt.subplot(1, 2, 2)
        bar_labels = class_names if class_names else [f"Class {i}" for i in range(len(prediction))]
        bars = plt.bar(bar_labels, prediction)
        
        # Tô màu thanh được dự đoán
        bars[predicted_class_idx].set_color('red')
        
        plt.title('Prediction Probabilities')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Lưu biểu đồ nếu yêu cầu
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    return result