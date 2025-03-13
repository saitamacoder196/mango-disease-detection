# File: src/training/evaluate.py
# Đánh giá mô hình

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
from src.data.dataloader import create_data_generators

def evaluate_model(model_path, test_dir, batch_size=32, img_size=(224, 224), evaluation_config=None):
    """Đánh giá mô hình trên tập kiểm tra."""
    
    if evaluation_config is None:
        evaluation_config = {}
    
    # Lấy cấu hình hoặc sử dụng giá trị mặc định
    save_dir = evaluation_config.get('save_dir', 'evaluation_results')
    
    # Tạo thư mục lưu kết quả
    os.makedirs(save_dir, exist_ok=True)
    
    # Tải mô hình
    print(f"Loading model from {model_path}")
    model = load_model(model_path)
    
    # Tạo data generator cho tập kiểm tra
    _, _, test_generator = create_data_generators(
        train_dir=None,
        validation_dir=None,
        test_dir=test_dir,
        img_size=img_size,
        batch_size=batch_size,
        augmentation=False
    )
    
    # Đánh giá mô hình
    print("Evaluating model...")
    evaluation = model.evaluate(test_generator)
    
    # In kết quả đánh giá
    print(f"Test Loss: {evaluation[0]:.4f}")
    print(f"Test Accuracy: {evaluation[1]:.4f}")
    
    # Dự đoán trên tập kiểm tra
    print("Generating predictions...")
    test_generator.reset()
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Lấy nhãn thực tế
    y_true = test_generator.classes
    
    # Lấy tên các lớp
    class_names = list(test_generator.class_indices.keys())
    
    # Tạo classification report
    report = classification_report(y_true, y_pred_classes, target_names=class_names, output_dict=True)
    print("Classification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))
    
    # Lưu classification report vào file
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(classification_report(y_true, y_pred_classes, target_names=class_names))
    
    # Tạo và lưu confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    save_confusion_matrix(cm, class_names, os.path.join(save_dir, 'confusion_matrix.png'))
    
    # Tạo và lưu ROC curve
    save_roc_curve(y_true, y_pred, class_names, os.path.join(save_dir, 'roc_curve.png'))
    
    # Hiển thị một số ảnh dự đoán
    visualize_predictions(model, test_generator, class_names, save_path=os.path.join(save_dir, 'sample_predictions.png'))
    
    return evaluation, report

def save_confusion_matrix(cm, class_names, save_path):
    """Tạo và lưu ma trận nhầm lẫn."""
    plt.figure(figsize=(10, 8))
    
    # Tính normalization
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Vẽ heatmap
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title('Confusion Matrix')
    
    # Lưu biểu đồ
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_roc_curve(y_true, y_pred, class_names, save_path):
    """Tạo và lưu ROC curve."""
    plt.figure(figsize=(10, 8))
    
    # Chuyển đổi y_true sang one-hot encoding
    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=len(class_names))
    
    # Tính ROC curve và AUC cho từng lớp
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Vẽ ROC curve cho từng lớp
        plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    # Vẽ đường chéo
    plt.plot([0, 1], [0, 1], 'k--')
    
    # Cấu hình biểu đồ
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    
    # Lưu biểu đồ
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_predictions(model, test_generator, class_names, num_samples=5, save_path=None):
    """Hiển thị một số mẫu dự đoán."""
    plt.figure(figsize=(15, 3 * num_samples))
    
    # Lấy một batch dữ liệu
    test_generator.reset()
    x_batch, y_batch = next(test_generator)
    
    # Dự đoán
    y_pred = model.predict(x_batch)
    
    # Hiển thị kết quả
    for i in range(min(num_samples, len(x_batch))):
        # Hiển thị ảnh
        plt.subplot(num_samples, 2, 2*i+1)
        plt.imshow(x_batch[i])
        true_class = np.argmax(y_batch[i])
        plt.title(f"True: {class_names[true_class]}")
        plt.axis('off')
        
        # Hiển thị biểu đồ dự đoán
        plt.subplot(num_samples, 2, 2*i+2)
        pred_class = np.argmax(y_pred[i])
        plt.bar(range(len(class_names)), y_pred[i])
        plt.xticks(range(len(class_names)), class_names, rotation=90)
        plt.title(f"Pred: {class_names[pred_class]}")
    
    # Lưu biểu đồ nếu yêu cầu
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()