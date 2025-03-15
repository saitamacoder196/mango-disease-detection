# analyze_model.py
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model

# Định nghĩa các metrics tùy chỉnh để tương thích với mô hình đã lưu
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

# Tên các lớp
CLASS_NAMES = ["background", "da_cam", "da_ech", "dom_den", "than_thu", "rui_dut"]

# Tải mô hình
model_path = 'models/unet_model.h5'
try:
    print(f"Đang tải mô hình từ {model_path}...")
    model = load_model(
        model_path,
        custom_objects={
            'iou_score': iou_score,
            'f1_score': f1_score,
            'f1-score': f1_score
        }
    )
    print("Đã tải mô hình thành công")
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Tải một ảnh mẫu từ tập test
test_img_path = 'data/segmentation/test/images/3-XL-TT-Center_1.jpg'  # Thay bằng ảnh bệnh đã biết
if not os.path.exists(test_img_path):
    print(f"Không tìm thấy ảnh tại {test_img_path}")
    # Tìm một ảnh khác trong thư mục test
    test_dir = 'data/segmentation/test/images'
    if os.path.exists(test_dir):
        img_files = os.listdir(test_dir)
        if img_files:
            test_img_path = os.path.join(test_dir, img_files[0])
            print(f"Sử dụng ảnh thay thế: {test_img_path}")
        else:
            print("Không tìm thấy ảnh nào trong thư mục test")
            exit(1)
    else:
        print(f"Không tìm thấy thư mục test: {test_dir}")
        exit(1)

# Đọc và xử lý ảnh
print(f"Đang đọc ảnh từ {test_img_path}...")
img = cv2.imread(test_img_path)
if img is None:
    print(f"Không thể đọc ảnh từ {test_img_path}")
    exit(1)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, (512, 512))
img_input = img_resized / 255.0
img_input = np.expand_dims(img_input, axis=0)

# Dự đoán
print("Đang thực hiện dự đoán...")
pred = model.predict(img_input)[0]

# Phân tích dự đoán
print("Hình dạng dự đoán:", pred.shape)
print("Giá trị dự đoán nhỏ nhất:", np.min(pred))
print("Giá trị dự đoán lớn nhất:", np.max(pred))
print("Lớp có dự đoán cao nhất:", np.argmax(pred, axis=-1).max())

# Xem phân phối giá trị dự đoán cho từng lớp
for i in range(pred.shape[-1]):
    values = pred[:,:,i].flatten()
    mean_val = np.mean(values)
    max_val = np.max(values)
    print(f"Lớp {i} ({CLASS_NAMES[i]}): Trung bình={mean_val:.6f}, Tối đa={max_val:.6f}")

# Vẽ phân phối xác suất cho mỗi lớp
plt.figure(figsize=(15, 10))
for i in range(pred.shape[-1]):
    plt.subplot(2, 3, i+1)
    plt.imshow(pred[:,:,i], cmap='jet', vmin=0, vmax=1)
    plt.colorbar()
    plt.title(f"Xác suất Lớp {i} ({CLASS_NAMES[i]})")
plt.tight_layout()
plt.savefig('prediction_analysis.png')

# Vẽ mask dự đoán
pred_mask = np.argmax(pred, axis=-1)
plt.figure(figsize=(10, 10))
plt.imshow(pred_mask, cmap='viridis')
plt.colorbar()
plt.title("Mask dự đoán (argmax)")
plt.savefig('predicted_mask.png')

# Vẽ ảnh gốc bên cạnh mask
plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.imshow(img_resized)
plt.title("Ảnh gốc")
plt.subplot(1, 2, 2)
plt.imshow(pred_mask, cmap='viridis')
plt.title("Mask dự đoán")
plt.savefig('image_vs_mask.png')

print("\nPhân tích hoàn tất. Kiểm tra các file sau:")
print("- prediction_analysis.png: Xác suất của từng lớp")
print("- predicted_mask.png: Mask dự đoán")
print("- image_vs_mask.png: So sánh ảnh gốc và mask")