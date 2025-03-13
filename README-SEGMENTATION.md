# Hướng dẫn sử dụng Mô hình Phân đoạn Bệnh trên Xoài

Dự án này đã được mở rộng để hỗ trợ phân đoạn (segmentation) các bệnh trên da xoài, bao gồm: Da cám (DC), Da ếch (DE), Đóm đen (DD), Thán thư (TT), và Rùi đụt (RD).

## 1. Chuẩn bị dữ liệu

### 1.1. Cấu trúc dữ liệu nguồn

Dữ liệu nguồn có cấu trúc như sau:
```
folder_gốc/
├── dot1/
│   ├── thư_mục_con_1/
│   ├── thư_mục_con_2/
│   └── ...
├── dot2/
│   ├── thư_mục_con_1/
│   ├── thư_mục_con_2/
│   └── ...
└── ...
```

Trong mỗi thư mục con chứa:
- Các file ảnh `.jpg`
- Các file annotation `.json` tương ứng (định dạng Labelme)

### 1.2. Xử lý dữ liệu

Sử dụng script `prepare_segmentation_data.py` để xử lý dữ liệu:

```bash
python scripts/prepare_segmentation_data.py --input_dir <đường_dẫn_thư_mục_gốc> --output_dir data
```

Các tham số:
- `--input_dir`: Đường dẫn đến thư mục dữ liệu gốc (bắt buộc)
- `--output_dir`: Thư mục đầu ra (mặc định: "data")
- `--img_size`: Kích thước ảnh đầu ra (mặc định: 512 512)
- `--val_split`: Tỷ lệ dữ liệu validation (mặc định: 0.15)
- `--test_split`: Tỷ lệ dữ liệu test (mặc định: 0.15)

## 2. Cài đặt môi trường

Trước khi sử dụng mô hình phân đoạn, cần cài đặt các thư viện bổ sung. Sử dụng script:

```bash
python install_segmentation_requirements.py
```

Script sẽ cung cấp hai lựa chọn:
1. Cài đặt các thư viện vào môi trường hiện tại
2. Tạo môi trường Conda mới và cài đặt thư viện

Các thư viện cần thiết bao gồm:
- segmentation-models
- albumentations
- tensorflow (>=2.4.0)
- opencv-python
- scikit-learn
- matplotlib
- pyyaml
- tqdm

## 3. Cấu hình mô hình

File cấu hình mô hình nằm tại `configs/segmentation_config.yaml`. Các tham số quan trọng:

```yaml
# Cấu hình dữ liệu
data:
  train_dir: "data/segmentation/train"
  validation_dir: "data/segmentation/val"
  test_dir: "data/segmentation/test"
  img_size: [512, 512]

# Cấu hình mô hình
model:
  num_classes: 6  # 5 loại bệnh + 1 background
  class_names: ["background", "da_cam", "da_ech", "dom_den", "than_thu", "rui_dut"]
  class_mapping: {"background": 0, "DC": 1, "DE": 2, "DD": 3, "TT": 4, "RD": 5}
  
  # Cấu hình riêng cho mô hình segmentation
  segmentation_model:
    architecture: "unet"  # Lựa chọn: 'unet', 'fpn', 'pspnet', 'deeplabv3', 'linknet'
    encoder: "resnet34"   # Encoder backbone: 'resnet34', 'resnet50', 'efficientnetb0', 'mobilenetv2'
```

## 4. Huấn luyện mô hình

Huấn luyện mô hình phân đoạn bằng lệnh:

```bash
python segmentation_main.py --mode train --config configs/segmentation_config.yaml
```

Mô hình sẽ được lưu tại `models/segmentation_model.h5`

## 5. Đánh giá mô hình

Đánh giá mô hình trên tập test:

```bash
python segmentation_main.py --mode evaluate --config configs/segmentation_config.yaml --model_path models/segmentation_model.h5
```

Kết quả đánh giá sẽ được lưu tại `evaluation_results/segmentation/`

## 6. Dự đoán trên ảnh mới

Sử dụng mô hình đã huấn luyện để dự đoán trên ảnh mới:

```bash
python segmentation_main.py --mode predict --config configs/segmentation_config.yaml --model_path models/segmentation_model.h5 --image_path path/to/your/image.jpg --output_path results/prediction.png
```

## 7. Giải thích kết quả

Kết quả phân đoạn sẽ có mã màu như sau:
- Đen: Background (không có bệnh)
- Đỏ: Da cám (DC)
- Xanh lá: Da ếch (DE)
- Xanh dương: Đóm đen (DD)
- Vàng: Thán thư (TT)
- Tím: Rùi đụt (RD)

## 8. Tích hợp với web app hiện tại

Để tích hợp phân đoạn bệnh vào web app hiện tại, cần cập nhật file `app/routes.py` để thêm route xử lý phân đoạn. Chi tiết cập nhật có trong file `app/routes_segmentation.py`.

## 9. Các lưu ý khi sử dụng

1. Dữ liệu phân đoạn đòi hỏi nhiều tài nguyên hơn so với phân loại đơn thuần
2. Nên sử dụng GPU để tăng tốc độ huấn luyện và dự đoán
3. Kích thước ảnh đầu vào ảnh hưởng nhiều đến hiệu năng mô hình và tài nguyên cần thiết
4. Nên thử nghiệm nhiều kiến trúc mô hình khác nhau (unet, fpn, pspnet, deeplabv3)

## 10. Các cải tiến trong tương lai

1. Thêm mô hình hậu xử lý để tăng chất lượng phân đoạn
2. Tối ưu hóa mô hình cho thiết bị di động
3. Thêm khả năng phát hiện vùng nhiễm bệnh với nhiều mức độ (nhẹ, trung bình, nặng)
4. Kết hợp phân đoạn với phân loại để cải thiện độ chính xác