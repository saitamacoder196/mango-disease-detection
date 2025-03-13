# File: app/routes_segmentation.py
# Mở rộng routes cho web app để hỗ trợ phân đoạn bệnh trên xoài

import os
import numpy as np
from flask import render_template, request, jsonify, url_for, send_file
from werkzeug.utils import secure_filename
import cv2
import tensorflow as tf
import yaml
import segmentation_models as sm
from app import app

# Cấu hình đường dẫn lưu ảnh tải lên cho phân đoạn
SEGMENTATION_UPLOAD_FOLDER = 'app/static/segmentation_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Tạo thư mục nếu chưa tồn tại
os.makedirs(SEGMENTATION_UPLOAD_FOLDER, exist_ok=True)

# Thông tin mô hình phân đoạn
SEGMENTATION_MODEL_PATH = 'models/segmentation_model.h5'
CONFIG_PATH = 'configs/segmentation_config.yaml'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_segmentation_model():
    """Tải mô hình phân đoạn."""
    model = tf.keras.models.load_model(
        SEGMENTATION_MODEL_PATH,
        custom_objects={
            'iou_score': sm.metrics.IOUScore(threshold=0.5),
            'f1-score': sm.metrics.FScore(threshold=0.5)
        }
    )
    return model

def load_config():
    """Tải file cấu hình."""
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    return config

def segment_image(img_path, model, config):
    """Thực hiện phân đoạn ảnh."""
    model_config = config['model']
    img_size = tuple(model_config['input_shape'][:2])
    
    # Đọc ảnh
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize ảnh
    original_img = img.copy()
    img_resized = cv2.resize(img, img_size)
    
    # Chuẩn bị đầu vào
    img_input = img_resized / 255.0
    img_input = np.expand_dims(img_input, axis=0)
    
    # Dự đoán
    pred = model.predict(img_input)[0]
    pred_mask = np.argmax(pred, axis=-1)
    
    # Màu cho các lớp (BGR cho OpenCV)
    colors = [
        [0, 0, 0],      # Background - đen
        [0, 0, 255],    # Da cám - đỏ
        [0, 255, 0],    # Da ếch - xanh lá
        [255, 0, 0],    # Đóm đen - xanh dương
        [0, 255, 255],  # Thán thư - vàng
        [255, 0, 255]   # Rùi đụt - tím
    ]
    
    # Tạo mask màu
    colored_mask = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    for class_idx, color in enumerate(colors):
        colored_mask[pred_mask == class_idx] = color
    
    # Tạo overlay
    alpha = 0.6
    overlay_img = cv2.addWeighted(img_resized, 1-alpha, colored_mask, alpha, 0)
    
    # Tính tỷ lệ diện tích từng loại bệnh
    total_pixels = pred_mask.size
    class_areas = {}
    class_names = model_config['class_names']
    
    for class_idx, class_name in enumerate(class_names):
        pixel_count = np.sum(pred_mask == class_idx)
        percentage = (pixel_count / total_pixels) * 100
        class_areas[class_name] = percentage
    
    # Kết quả trả về
    result = {
        'mask': cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR),
        'overlay': cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR),
        'class_areas': class_areas
    }
    
    return result

@app.route('/segmentation')
def segmentation_page():
    """Trang phân đoạn bệnh."""
    return render_template('segmentation.html')

@app.route('/upload_segmentation', methods=['POST'])
def upload_segmentation():
    """Xử lý tải ảnh lên và phân đoạn."""
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file nào được tải lên'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Không có file nào được chọn'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(SEGMENTATION_UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        try:
            # Tải mô hình và cấu hình
            model = load_segmentation_model()
            config = load_config()
            
            # Thực hiện phân đoạn
            result = segment_image(file_path, model, config)
            
            # Lưu ảnh kết quả
            mask_filename = f'mask_{filename}'
            overlay_filename = f'overlay_{filename}'
            mask_path = os.path.join(SEGMENTATION_UPLOAD_FOLDER, mask_filename)
            overlay_path = os.path.join(SEGMENTATION_UPLOAD_FOLDER, overlay_filename)
            
            cv2.imwrite(mask_path, result['mask'])
            cv2.imwrite(overlay_path, result['overlay'])
            
            # Chuẩn bị kết quả trả về
            response = {
                'original_image': url_for('static', filename=f'segmentation_uploads/{filename}'),
                'mask_image': url_for('static', filename=f'segmentation_uploads/{mask_filename}'),
                'overlay_image': url_for('static', filename=f'segmentation_uploads/{overlay_filename}'),
                'class_areas': result['class_areas']
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({'error': f'Lỗi khi phân đoạn ảnh: {str(e)}'})
    
    return jsonify({'error': 'File không được chấp nhận'})

@app.route('/segmentation_info')
def segmentation_info():
    """Thông tin về phân đoạn bệnh trên da xoài."""
    disease_info = {
        'background': {
            'name': 'Vùng không bệnh',
            'description': 'Phần da xoài không bị nhiễm bệnh.',
            'color': '#000000'  # Đen
        },
        'da_cam': {
            'name': 'Da cám (DC)',
            'scientific_name': 'Bệnh do nấm Colletotrichum gloeosporioides',
            'description': 'Bệnh da cám làm cho vỏ quả xoài trở nên sần sùi, mất giá trị thương phẩm.',
            'symptoms': [
                'Vỏ quả xuất hiện các chấm nhỏ màu nâu hoặc đen',
                'Các chấm nhỏ lan rộng và hợp lại với nhau',
                'Bề mặt quả sần sùi, khô cứng',
                'Hình thành các lớp vảy nhỏ trên vỏ quả'
            ],
            'treatment': [
                'Phun thuốc phòng trừ nấm định kỳ',
                'Cắt tỉa cành để thông thoáng',
                'Thu gom và tiêu hủy quả bị bệnh',
                'Bón phân cân đối NPK'
            ],
            'color': '#FF0000'  # Đỏ
        },
        'da_ech': {
            'name': 'Da ếch (DE)',
            'scientific_name': 'Bệnh do nấm và vi khuẩn kết hợp',
            'description': 'Bệnh da ếch làm cho vỏ quả xoài có màu xanh đậm và sần sùi như da ếch.',
            'symptoms': [
                'Vỏ quả có màu xanh đậm bất thường',
                'Bề mặt quả sần sùi, không đều',
                'Vỏ quả dày và cứng hơn bình thường',
                'Vỏ quả không chuyển màu khi chín'
            ],
            'treatment': [
                'Phun thuốc phòng trừ nấm và vi khuẩn',
                'Bảo đảm độ ẩm phù hợp trong vườn',
                'Cắt tỉa cành thường xuyên',
                'Bọc quả trong giai đoạn phát triển'
            ],
            'color': '#00FF00'  # Xanh lá
        },
        'dom_den': {
            'name': 'Đóm đen (DD)',
            'scientific_name': 'Bệnh do nấm Alternaria alternata',
            'description': 'Bệnh đóm đen gây ra các đốm đen trên vỏ quả xoài, làm giảm chất lượng quả.',
            'symptoms': [
                'Xuất hiện các đốm đen nhỏ trên vỏ quả',
                'Các đốm đen có viền nâu hoặc vàng xung quanh',
                'Các đốm có thể hợp nhất thành mảng lớn',
                'Đốm đen có thể lõm xuống so với bề mặt quả'
            ],
            'treatment': [
                'Phun thuốc đặc hiệu phòng trừ nấm Alternaria',
                'Tránh làm tổn thương quả khi thu hoạch',
                'Duy trì khoảng cách giữa các quả',
                'Đảm bảo thông gió tốt trong vườn'
            ],
            'color': '#0000FF'  # Xanh dương
        },
        'than_thu': {
            'name': 'Thán thư (TT)',
            'scientific_name': 'Colletotrichum gloeosporioides',
            'description': 'Bệnh thán thư là một trong những bệnh phổ biến nhất trên xoài, gây ra các vết đốm đen hoặc nâu trên lá, hoa và quả.',
            'symptoms': [
                'Vết đốm đen hoặc nâu trên lá và quả',
                'Vết bệnh có hình tròn hoặc bầu dục',
                'Quả bị bệnh có thể bị thối và rụng sớm',
                'Hoa bị nhiễm bệnh có thể bị khô và rụng'
            ],
            'treatment': [
                'Loại bỏ và tiêu hủy các bộ phận bị nhiễm bệnh',
                'Sử dụng thuốc diệt nấm có chứa copper oxychloride hoặc mancozeb',
                'Áp dụng biện pháp vệ sinh vườn cây',
                'Phun thuốc phòng ngừa trong mùa mưa'
            ],
            'color': '#FFFF00'  # Vàng
        },
        'rui_dut': {
            'name': 'Rùi đụt (RD)',
            'scientific_name': 'Bệnh do một số loài nấm',
            'description': 'Bệnh rùi đụt gây ra các vết thối nhỏ trên vỏ quả, làm giảm giá trị thương phẩm.',
            'symptoms': [
                'Vỏ quả xuất hiện các vết thối nhỏ',
                'Vết thối có màu nâu đến đen',
                'Vết thối có thể lõm xuống',
                'Phần thịt quả bên dưới vết thối có thể bị ảnh hưởng'
            ],
            'treatment': [
                'Phun thuốc phòng trừ nấm định kỳ',
                'Thu hoạch quả đúng thời điểm',
                'Bảo quản quả ở nhiệt độ và độ ẩm phù hợp',
                'Xử lý quả sau thu hoạch bằng nước nóng'
            ],
            'color': '#FF00FF'  # Tím
        }
    }
    
    return render_template('segmentation_info.html', disease_info=disease_info)