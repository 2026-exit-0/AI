# issue 기록 남기기 주석(dacb7239103f0bffa4edd4fd7d4c4685bd795e55)
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision import transforms as T

# 1. 설정
DATA_DIR = './processed_data' # 전처리된 결과가 있는 곳
SAMPLE_SIZE = 4

labels = ['D', 'P', 'T']

def verify_samples():
    # 라벨 폴더 중 하나(예: D)에서 샘플 가져오기
    # Resnet-50의 경우 Mean이 0.4~6정도 나와야함
    for label in labels:
        folder_path = os.path.join(DATA_DIR, label)
    
        if not os.path.exists(folder_path):
            print(f"폴더가 없습니다: {folder_path}")
            return

        files = [f for f in os.listdir(folder_path)][:SAMPLE_SIZE]
    
        plt.figure(figsize=(12, 4))
    
        for i, f in enumerate(files):
            # 전처리된 파일은 이미 텐서 저장이 아닌 이미지로 저장되었을 것이므로 다시 읽기
            img_path = os.path.join(folder_path, f)
            img = Image.open(img_path)
        
            plt.subplot(1, SAMPLE_SIZE, i+1)
            plt.imshow(img)

            # 파일명에서 부위 이름(예: Forehead)만 추출해서 제목으로 표시
            part_name = f.split('_')[0] 
            plt.title(f"{part_name}") 
            plt.axis('off')
        
            # 데이터 통계 확인을 위해 임시로 텐서 변환
            temp_tensor = T.ToTensor()(img)
            print(f"Sample {i+1} - Mean: {temp_tensor.mean():.4f}, Shape: {temp_tensor.shape}")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    verify_samples()