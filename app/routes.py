# File: app/routes.py
# Định nghĩa routes cho ứng dụng web demo

import os
import numpy as np
from flask import render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from app import app
from src.inference.predict import predict_image
from src.inference.gradcam import display_gradcam

# Cấu hình đường dẫn lưu ảnh tải lên
UPLOAD_FOLDER = 'app/static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Mô hình và thông tin liên quan
MODEL_PATH = 'models/mobilenetv2_model.h5'
LAST_CONV_LAYER = 'Conv_1'  # Tên lớp tích chập cuối cùng của MobileNetV2
CLASS_NAMES = ["anthracnose", "bacterial_canker", "healthy", "powdery_mildew"]
IMG_SIZE = (224, 224)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Trang chủ."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Xử lý tải ảnh lên và dự đoán."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Dự đoán
        result = predict_image(
            model_path=MODEL_PATH,
            image_path=file_path,
            img_size=IMG_SIZE,
            class_names=CLASS_NAMES,
            visualize=True,
            save_path=os.path.join(app.config['UPLOAD_FOLDER'], 'prediction_' + filename)
        )
        
        # Tạo Grad-CAM
        display_gradcam(
            img_path=file_path,
            model_path=MODEL_PATH,
            last_conv_layer_name=LAST_CONV_LAYER,
            class_names=CLASS_NAMES,
            save_path=os.path.join(app.config['UPLOAD_FOLDER'], 'gradcam_' + filename)
        )
        
        # Chuẩn bị kết quả trả về
        response = {
            'prediction': result['class'],
            'confidence': f"{result['confidence']*100:.2f}%",
            'original_image': url_for('static', filename=f'uploads/{filename}'),
            'prediction_image': url_for('static', filename=f'uploads/prediction_{filename}'),
            'gradcam_image': url_for('static', filename=f'uploads/gradcam_{filename}'),
            'probabilities': result['probabilities']
        }
        
        return jsonify(response)
    
    return jsonify({'error': 'File not allowed'})

@app.route('/info')
def info():
    """Thông tin về các loại bệnh trên da xoài."""
    disease_info = {
        'anthracnose': {
            'name': 'Bệnh thán thư',
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
            ]
        },
        'bacterial_canker': {
            'name': 'Bệnh loét vi khuẩn',
            'scientific_name': 'Xanthomonas campestris pv. mangiferaeindicae',
            'description': 'Bệnh loét vi khuẩn gây ra các vết loét trên lá, cành và quả xoài, đặc biệt nghiêm trọng trong điều kiện ẩm ướt.',
            'symptoms': [
                'Vết loét nhỏ có viền nước trên lá và quả',
                'Vết loét có màu đen hoặc nâu đậm',
                'Tiết dịch trên cành bị nhiễm bệnh',
                'Lá bị rụng sớm, quả bị nứt'
            ],
            'treatment': [
                'Cắt tỉa và tiêu hủy các bộ phận bị nhiễm bệnh',
                'Sử dụng thuốc kháng sinh nông nghiệp như streptomycin',
                'Phun dung dịch copper oxychloride',
                'Tránh tưới nước từ trên cao xuống trong mùa mưa'
            ]
        },
        'powdery_mildew': {
            'name': 'Bệnh phấn trắng',
            'scientific_name': 'Oidium mangiferae',
            'description': 'Bệnh phấn trắng tạo ra lớp phủ màu trắng như bột trên lá, hoa và quả non, gây ảnh hưởng đến khả năng quang hợp và ra hoa.',
            'symptoms': [
                'Lớp phủ màu trắng như bột trên lá, hoa và quả non',
                'Lá bị biến dạng hoặc quăn lại',
                'Hoa bị khô và rụng',
                'Quả non bị rụng'
            ],
            'treatment': [
                'Phun lưu huỳnh hoặc các thuốc diệt nấm chuyên dụng',
                'Cắt tỉa để cải thiện lưu thông không khí',
                'Tránh tưới nước quá nhiều',
                'Áp dụng phun phòng ngừa trước mùa ra hoa'
            ]
        },
        'healthy': {
            'name': 'Cây khỏe mạnh',
            'description': 'Cây xoài khỏe mạnh có lá xanh tươi, quả phát triển bình thường và không có dấu hiệu của bệnh tật.',
            'characteristics': [
                'Lá xanh tươi và cứng cáp',
                'Quả phát triển đều và có màu sắc tự nhiên',
                'Không có vết đốm, loét hoặc phủ bột',
                'Cây sinh trưởng mạnh'
            ],
            'maintenance': [
                'Tưới nước đầy đủ nhưng tránh quá ẩm',
                'Bón phân cân đối NPK',
                'Cắt tỉa định kỳ để loại bỏ cành già, yếu',
                'Theo dõi thường xuyên để phát hiện sớm dấu hiệu bệnh tật'
            ]
        }
    }
    
    return render_template('info.html', disease_info=disease_info)