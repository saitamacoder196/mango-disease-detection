# File: src/inference/gradcam.py
# Hiển thị Grad-CAM (Gradient-weighted Class Activation Mapping)

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def get_img_array(img_path, size=(224, 224)):
    """Đọc và tiền xử lý ảnh."""
    # Đọc ảnh và chuyển sang RGB
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize ảnh
    img = cv2.resize(img, size)
    
    # Mở rộng kích thước batch và chuẩn hóa
    img_array = np.expand_dims(img, axis=0) / 255.0
    
    return img_array, img

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Tạo heatmap Grad-CAM."""
    # Lấy lớp tích chập cuối cùng và mô hình đầu ra
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Tính gradient của lớp đầu ra với respect to lớp tích chập cuối
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradient của lớp đầu ra với respect to feature map
    grads = tape.gradient(class_channel, conv_output)

    # Pooling gradients để lấy trọng số của các kênh
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Tính trung bình các feature maps với trọng số
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Chuẩn hóa heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def display_gradcam(img_path, model_path, last_conv_layer_name, class_names=None, save_path=None):
    """Hiển thị Grad-CAM cho ảnh."""
    # Tải mô hình
    model = load_model(model_path)
    
    # Đọc và tiền xử lý ảnh
    img_array, img = get_img_array(img_path)
    
    # Dự đoán
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    
    # Lấy tên lớp dự đoán
    if class_names:
        predicted_class = class_names[predicted_class_idx]
    else:
        predicted_class = f"Class {predicted_class_idx}"
    
    # Tạo heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=predicted_class_idx)
    
    # Resize heatmap về kích thước ảnh gốc
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Chuyển heatmap về RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Kết hợp heatmap với ảnh gốc
    superimposed_img = heatmap * 0.4 + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    
    # Hiển thị kết quả
    plt.figure(figsize=(12, 5))
    
    # Ảnh gốc
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title(f"Original Image\nPrediction: {predicted_class}\nConfidence: {predictions[0][predicted_class_idx]:.2f}")
    plt.axis('off')
    
    # Heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')
    
    # Ảnh kết hợp
    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img)
    plt.title("Superimposed Image")
    plt.axis('off')
    
    plt.tight_layout()
    
    # Lưu hình nếu yêu cầu
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    
    return predicted_class, predictions[0][predicted_class_idx]