#!/usr/bin/env python
# Script cài đặt các thư viện bổ sung cho mô hình phân đoạn

import subprocess
import sys
import os

def install_requirements():
    """Cài đặt các thư viện cần thiết cho mô hình phân đoạn."""
    requirements = [
        "segmentation-models",
        "albumentations",
        "opencv-python",
        "tensorflow>=2.4.0",
        "scikit-learn",
        "matplotlib",
        "pyyaml",
        "tqdm"
    ]
    
    print("Đang cài đặt các thư viện cần thiết cho mô hình phân đoạn...")
    
    for req in requirements:
        print(f"Đang cài đặt {req}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
        except subprocess.CalledProcessError as e:
            print(f"Lỗi khi cài đặt {req}: {e}")
            
    print("\nĐã cài đặt xong các thư viện cần thiết!")
    
    # Kiểm tra cài đặt
    try:
        import segmentation_models as sm
        import albumentations as A
        import tensorflow as tf
        import cv2
        
        print("\nKiểm tra cài đặt:")
        print(f"TensorFlow phiên bản: {tf.__version__}")
        print(f"OpenCV phiên bản: {cv2.__version__}")
        print("Segmentation Models: Đã cài đặt")
        print("Albumentations: Đã cài đặt")
        
        # Kiểm tra GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"\nĐã phát hiện {len(gpus)} GPU:")
            for gpu in gpus:
                print(f"  - {gpu}")
        else:
            print("\nKhông phát hiện GPU. Mô hình sẽ chạy trên CPU.")
            print("Để tăng tốc độ huấn luyện, bạn nên cài đặt TensorFlow phiên bản GPU.")
        
    except ImportError as e:
        print(f"\nLỗi khi kiểm tra cài đặt: {e}")
        print("Vui lòng kiểm tra lại việc cài đặt các thư viện.")

def create_conda_env():
    """Tạo môi trường Conda mới với các thư viện cần thiết."""
    env_name = input("Nhập tên môi trường Conda mới (mặc định: mango-segmentation): ") or "mango-segmentation"
    python_ver = input("Nhập phiên bản Python (mặc định: 3.8): ") or "3.8"
    
    print(f"\nĐang tạo môi trường Conda '{env_name}' với Python {python_ver}...")
    
    try:
        # Tạo môi trường mới
        subprocess.check_call(["conda", "create", "-n", env_name, f"python={python_ver}", "-y"])
        
        # Lấy đường dẫn đến script activate
        if sys.platform == 'win32':
            activate_script = os.path.join(os.environ.get('CONDA_PREFIX'), 'Scripts', 'activate')
        else:
            activate_script = os.path.join(os.environ.get('CONDA_PREFIX'), 'bin', 'activate')
        
        # Cài đặt các thư viện cơ bản bằng conda
        subprocess.check_call(["conda", "install", "-n", env_name, "-c", "conda-forge", 
                               "tensorflow", "opencv", "scikit-learn", "matplotlib", 
                               "pyyaml", "tqdm", "-y"])
        
        # Kích hoạt môi trường và cài đặt các thư viện bằng pip
        install_cmd = f"conda activate {env_name} && pip install segmentation-models albumentations"
        if sys.platform == 'win32':
            subprocess.check_call(["cmd", "/c", f"call {activate_script} {env_name} && pip install segmentation-models albumentations"])
        else:
            subprocess.check_call(["bash", "-c", f"source {activate_script} {env_name} && pip install segmentation-models albumentations"])
        
        print(f"\nĐã tạo thành công môi trường '{env_name}'!")
        print(f"\nĐể kích hoạt môi trường, chạy lệnh:")
        print(f"conda activate {env_name}")
        
    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi tạo môi trường Conda: {e}")
        print("Vui lòng kiểm tra lại cài đặt Conda.")

if __name__ == "__main__":
    print("=== Cài đặt thư viện cho mô hình phân đoạn bệnh xoài ===\n")
    print("1. Cài đặt thư viện vào môi trường hiện tại")
    print("2. Tạo môi trường Conda mới và cài đặt thư viện")
    print("3. Thoát")
    
    choice = input("\nLựa chọn của bạn (1-3): ")
    
    if choice == '1':
        install_requirements()
    elif choice == '2':
        create_conda_env()
    else:
        print("Thoát chương trình.")