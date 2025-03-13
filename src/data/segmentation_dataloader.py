# File: src/data/segmentation_dataloader.py
# Bộ nạp dữ liệu cho mô hình phân đoạn

import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.utils import to_categorical
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, Rotate, ShiftScaleRotate,
    RandomBrightnessContrast, GaussianBlur, GaussianNoise,
    ElasticTransform, GridDistortion, OneOf
)


class SegmentationDataLoader:
    """Bộ nạp dữ liệu cho mô hình phân đoạn."""
    
    def __init__(self, 
                 data_dir, 
                 img_size=(512, 512), 
                 batch_size=8, 
                 num_classes=6,
                 augmentation=False, 
                 augmentation_config=None):
        """
        Khởi tạo bộ nạp dữ liệu.
        
        Args:
            data_dir: Thư mục chứa dữ liệu ('train', 'val', hoặc 'test')
            img_size: Kích thước ảnh đầu vào
            batch_size: Kích thước batch
            num_classes: Số lớp phân đoạn
            augmentation: Có áp dụng tăng cường dữ liệu hay không
            augmentation_config: Cấu hình tăng cường dữ liệu
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.augmentation = augmentation
        self.augmentation_config = augmentation_config or {}
        
        # Đường dẫn thư mục ảnh và mask
        self.images_dir = os.path.join(data_dir, 'images')
        self.masks_dir = os.path.join(data_dir, 'masks')
        
        # Lấy danh sách file
        self.image_files = sorted([f for f in os.listdir(self.images_dir) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        self.num_samples = len(self.image_files)
        self.indices = np.arange(self.num_samples)
        
        # Khởi tạo bộ tăng cường dữ liệu
        if self.augmentation:
            self.augmenter = self._create_augmenter()
        
        # Tính số batch mỗi epoch
        self.steps_per_epoch = self.num_samples // self.batch_size
        if self.num_samples % self.batch_size != 0:
            self.steps_per_epoch += 1
    
    def _create_augmenter(self):
        """Tạo bộ tăng cường dữ liệu dựa trên cấu hình."""
        aug_list = []
        
        # Thêm các phép biến đổi dựa trên cấu hình
        if self.augmentation_config.get('horizontal_flip', True):
            aug_list.append(HorizontalFlip(p=0.5))
        
        if self.augmentation_config.get('vertical_flip', False):
            aug_list.append(VerticalFlip(p=0.5))
        
        if self.augmentation_config.get('rotation', True):
            aug_list.append(Rotate(limit=self.augmentation_config.get('rotation_limit', 20), p=0.7))
        
        if self.augmentation_config.get('random_brightness_contrast', True):
            aug_list.append(RandomBrightnessContrast(
                brightness_limit=self.augmentation_config.get('brightness_limit', 0.2),
                contrast_limit=self.augmentation_config.get('contrast_limit', 0.2),
                p=0.7
            ))
        
        if self.augmentation_config.get('gaussian_blur', True):
            aug_list.append(GaussianBlur(
                blur_limit=self.augmentation_config.get('blur_limit', 7),
                p=0.3
            ))
        
        if self.augmentation_config.get('gaussian_noise', True):
            aug_list.append(GaussianNoise(
                var_limit=self.augmentation_config.get('noise_var_limit', (10.0, 50.0)),
                p=0.3
            ))
        
        if self.augmentation_config.get('shift', True) or self.augmentation_config.get('scale', True):
            aug_list.append(ShiftScaleRotate(
                shift_limit=self.augmentation_config.get('shift_limit', 0.1),
                scale_limit=self.augmentation_config.get('scale_limit', 0.2),
                rotate_limit=0,  # Không quay vì đã có Rotate ở trên
                p=0.7
            ))
        
        # Biến dạng nâng cao (tùy chọn)
        advanced_distortions = []
        
        if self.augmentation_config.get('elastic', True):
            advanced_distortions.append(ElasticTransform(
                alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5
            ))
        
        if self.augmentation_config.get('grid_distortion', True):
            advanced_distortions.append(GridDistortion(p=0.5))
        
        if advanced_distortions:
            aug_list.append(OneOf(advanced_distortions, p=0.3))
        
        return Compose(aug_list)
    
    def _load_image(self, image_path):
        """Đọc và xử lý ảnh."""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        return img
    
    def _load_mask(self, mask_path):
        """Đọc và xử lý mask."""
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        return mask
    
    def _one_hot_encode(self, mask):
        """Chuyển mask sang dạng one-hot."""
        return to_categorical(mask, self.num_classes)
    
    def __len__(self):
        """Trả về số batch mỗi epoch."""
        return self.steps_per_epoch
    
    def on_epoch_end(self):
        """Được gọi khi kết thúc mỗi epoch."""
        # Xáo trộn dữ liệu
        np.random.shuffle(self.indices)
    
    def __getitem__(self, idx):
        """Lấy batch thứ idx."""
        # Tính chỉ số bắt đầu và kết thúc của batch
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.num_samples)
        batch_indices = self.indices[start_idx:end_idx]
        
        # Khởi tạo batch
        batch_x = np.zeros((len(batch_indices), *self.img_size, 3), dtype=np.float32)
        batch_y = np.zeros((len(batch_indices), *self.img_size, self.num_classes), dtype=np.float32)
        
        # Đọc dữ liệu vào batch
        for i, idx in enumerate(batch_indices):
            image_file = self.image_files[idx]
            mask_file = os.path.splitext(image_file)[0] + '.png'  # Giả sử mask có đuôi .png
            
            image_path = os.path.join(self.images_dir, image_file)
            mask_path = os.path.join(self.masks_dir, mask_file)
            
            # Đọc ảnh và mask
            img = self._load_image(image_path)
            mask = self._load_mask(mask_path)
            
            # Tăng cường dữ liệu nếu cần
            if self.augmentation:
                augmented = self.augmenter(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
            
            # Chuẩn hóa ảnh và chuyển mask sang one-hot
            img = img / 255.0
            one_hot_mask = self._one_hot_encode(mask)
            
            # Lưu vào batch
            batch_x[i] = img
            batch_y[i] = one_hot_mask
        
        return batch_x, batch_y
    
    def create_tf_dataset(self):
        """Tạo tf.data.Dataset từ dữ liệu."""
        def generator():
            for i in range(len(self)):
                yield self[i]
        
        return tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(None, *self.img_size, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, *self.img_size, self.num_classes), dtype=tf.float32)
            )
        )

def create_segmentation_datasets(train_dir, validation_dir, test_dir=None, img_size=(512, 512), 
                               batch_size=8, num_classes=6, augmentation=True, augmentation_config=None):
    """
    Tạo các bộ dữ liệu cho huấn luyện, validation và test.
    
    Args:
        train_dir: Thư mục chứa dữ liệu huấn luyện
        validation_dir: Thư mục chứa dữ liệu validation
        test_dir: Thư mục chứa dữ liệu test (tùy chọn)
        img_size: Kích thước ảnh
        batch_size: Kích thước batch
        num_classes: Số lớp phân đoạn
        augmentation: Có áp dụng tăng cường dữ liệu hay không
        augmentation_config: Cấu hình tăng cường dữ liệu
        
    Returns:
        Các bộ dữ liệu train, validation và test (nếu có)
    """
    # Tạo bộ dữ liệu huấn luyện
    train_loader = SegmentationDataLoader(
        train_dir,
        img_size=img_size,
        batch_size=batch_size,
        num_classes=num_classes,
        augmentation=augmentation,
        augmentation_config=augmentation_config
    )
    
    train_dataset = train_loader.create_tf_dataset()
    
    # Tạo bộ dữ liệu validation
    val_loader = SegmentationDataLoader(
        validation_dir,
        img_size=img_size,
        batch_size=batch_size,
        num_classes=num_classes,
        augmentation=False  # Không tăng cường dữ liệu cho validation
    )
    
    val_dataset = val_loader.create_tf_dataset()
    
    # Tạo bộ dữ liệu test nếu có
    test_dataset = None
    if test_dir:
        test_loader = SegmentationDataLoader(
            test_dir,
            img_size=img_size,
            batch_size=batch_size,
            num_classes=num_classes,
            augmentation=False  # Không tăng cường dữ liệu cho test
        )
        
        test_dataset = test_loader.create_tf_dataset()
    
    return train_dataset, val_dataset, test_dataset, train_loader.steps_per_epoch, val_loader.steps_per_epoch