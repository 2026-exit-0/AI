import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision import transforms as T

# 1. 설정
DATA_DIR = './processed_data' # 전처리된 결과가 있는 곳
SAMPLE_SIZE = 4

def denormalize(tensor):
    """정규화된 텐서를 다시 시각화 가능한 이미지로 변환"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def verify_samples():
    # 라벨 폴더 중 하나(예: D)에서 샘플 가져오기
    # Resnet-50의 경우 Mean이 0.2~3정도 나와야함
    # D: 0.2대
    # P: 0.5~0.6
    # T: 약 0.4
    label = 'T'
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
        plt.title(f"Sample {i+1}")
        plt.axis('off')
        
        # 데이터 통계 확인을 위해 임시로 텐서 변환
        temp_tensor = T.ToTensor()(img)
        print(f"Sample {i+1} - Mean: {temp_tensor.mean():.4f}, Shape: {temp_tensor.shape}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    verify_samples()