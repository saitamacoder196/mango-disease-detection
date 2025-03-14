# Tạo file kiểm tra: check_sm.py
import segmentation_models as sm
print("Segmentation Models version:", sm.__version__)

# Kiểm tra các tham số 
import inspect
print("\nUnet parameters:")
print(inspect.signature(sm.Unet))

print("\nBackbone options:")
print(sm.backbones.get_available_backbone_names())