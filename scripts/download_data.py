# File: scripts/download_data.py
# Script tải dữ liệu từ nguồn bên ngoài

import os
import requests
import zipfile
from tqdm import tqdm
import argparse

def download_file(url, destination):
    """Tải file từ URL."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    # Tạo thanh tiến trình
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    # Lưu file
    with open(destination, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    
    progress_bar.close()

def extract_zip(zip_path, extract_to):
    """Giải nén file zip."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Lấy tổng số file trong zip
        total_files = len(zip_ref.infolist())
        
        # Tạo thanh tiến trình
        with tqdm(total=total_files, desc="Extracting") as pbar:
            for file in zip_ref.infolist():
                zip_ref.extract(file, extract_to)
                pbar.update(1)

def main():
    parser = argparse.ArgumentParser(description='Download mango disease dataset')
    parser.add_argument('--url', type=str, required=True, help='URL to download the dataset')
    parser.add_argument('--output_dir', type=str, default='data/raw', help='Directory to save the dataset')
    
    args = parser.parse_args()
    
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Tên file tạm thời
    temp_zip = os.path.join(args.output_dir, 'temp_dataset.zip')
    
    # Tải dữ liệu
    print(f"Downloading dataset from {args.url}...")
    download_file(args.url, temp_zip)
    
    # Giải nén
    print(f"Extracting dataset to {args.output_dir}...")
    extract_zip(temp_zip, args.output_dir)
    
    # Xóa file tạm
    os.remove(temp_zip)
    
    print("Dataset downloaded and extracted successfully!")

if __name__ == "__main__":
    main()