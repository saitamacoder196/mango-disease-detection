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
    """Xây dựng mô hình phân đoạn sử dụng thư viện Segmentation Models."""
    
    # Thiết lập framework cho segmentation-models
    sm.set_framework('tf.keras')
    
    # Tạo từ điển tham số cơ bản (không bao gồm tham số gây lỗi)
    params = {
        'classes': num_classes,
        'activation': activation,
    }
    
    # Thử thêm các tham số khác nếu phiên bản hỗ trợ
    try:
        if hasattr(sm.Unet, '__code__') and 'input_shape' in sm.Unet.__code__.co_varnames:
            params['input_shape'] = input_shape
        
        if hasattr(sm.Unet, '__code__') and 'encoder_weights' in sm.Unet.__code__.co_varnames:
            params['encoder_weights'] = encoder_weights
        
        # Thử thêm tham số encoder với nhiều tên khác nhau
        for encoder_param in ['encoder', 'backbone', 'encoder_name']:
            try:
                temp_params = params.copy()
                temp_params[encoder_param] = encoder
                model = getattr(sm, architecture.capitalize())(**temp_params)
                print(f"Thành công với tham số: {encoder_param}")
                return model
            except Exception as e:
                print(f"Thử với {encoder_param}: {e}")
                continue
                
        # Nếu không thành công với các tên tham số encoder, thử cách đơn giản nhất
        return getattr(sm, architecture.capitalize())(classes=num_classes, activation=activation)
        
    except Exception as e:
        print(f"Lỗi khi xây dựng mô hình: {e}")
        # Fallback: Thử cách tiếp cận đơn giản nhất
        if architecture.lower() == 'unet':
            return sm.Unet(classes=num_classes, activation=activation)
        elif architecture.lower() == 'fpn':
            return sm.FPN(classes=num_classes, activation=activation)
        elif architecture.lower() == 'pspnet':
            return sm.PSPNet(classes=num_classes, activation=activation)
        else:
            raise ValueError(f"Kiến trúc {architecture} không được hỗ trợ")

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