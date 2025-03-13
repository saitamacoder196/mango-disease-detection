# File: src/models/segmentation_model.py
# Mô hình phân đoạn cho bệnh xoài

import tensorflow as tf
import segmentation_models as sm
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

# Cấu hình các metric cho segmentation
sm.set_framework('tf.keras')
sm.framework()

def build_segmentation_model(input_shape=(512, 512, 3), 
                           num_classes=6, 
                           architecture='unet', 
                           encoder='resnet34', 
                           encoder_weights='imagenet',
                           activation='softmax'):
    """
    Xây dựng mô hình phân đoạn sử dụng thư viện Segmentation Models.
    
    Args:
        input_shape: Kích thước ảnh đầu vào (height, width, channels)
        num_classes: Số lớp phân đoạn
        architecture: Kiến trúc mô hình ('unet', 'fpn', 'pspnet', 'linknet', 'deeplabv3')
        encoder: Encoder backbone 
        encoder_weights: Trọng số khởi tạo cho encoder
        activation: Hàm kích hoạt đầu ra ('softmax', 'sigmoid')
        
    Returns:
        Model: Mô hình phân đoạn
    """
    # Chọn kiến trúc mô hình
    if architecture.lower() == 'unet':
        model_fn = sm.Unet
    elif architecture.lower() == 'fpn':
        model_fn = sm.FPN
    elif architecture.lower() == 'pspnet':
        model_fn = sm.PSPNet
    elif architecture.lower() == 'linknet':
        model_fn = sm.Linknet
    elif architecture.lower() == 'deeplabv3':
        model_fn = sm.DeepLabV3
    else:
        raise ValueError(f"Kiến trúc {architecture} không được hỗ trợ")
    
    # Xây dựng mô hình
    model = model_fn(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        classes=num_classes,
        activation=activation,
        input_shape=input_shape
    )
    
    return model

def get_segmentation_metrics():
    """
    Trả về các metrics phù hợp cho mô hình phân đoạn.
    
    Returns:
        list: Danh sách các metrics
    """
    return [
        sm.metrics.IOUScore(threshold=0.5),  # IoU score
        sm.metrics.FScore(threshold=0.5),    # F1 score
        'accuracy'
    ]

def get_segmentation_loss(loss_name='categorical_crossentropy', class_weights=None):
    """
    Trả về hàm loss phù hợp cho mô hình phân đoạn.
    
    Args:
        loss_name: Tên của hàm loss ('categorical_crossentropy', 'dice_loss', 'focal_loss', 'jaccard_loss')
        class_weights: Trọng số cho các lớp
        
    Returns:
        loss: Hàm loss
    """
    if loss_name == 'categorical_crossentropy':
        return 'categorical_crossentropy'
    elif loss_name == 'dice_loss':
        return sm.losses.DiceLoss(class_weights=class_weights)
    elif loss_name == 'focal_loss':
        return sm.losses.CategoricalFocalLoss()
    elif loss_name == 'jaccard_loss':
        return sm.losses.JaccardLoss(class_weights=class_weights)
    elif loss_name == 'combined_loss':
        dice_loss = sm.losses.DiceLoss(class_weights=class_weights)
        focal_loss = sm.losses.CategoricalFocalLoss()
        return dice_loss + focal_loss
    else:
        raise ValueError(f"Loss {loss_name} không được hỗ trợ")