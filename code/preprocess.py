import os
import shutil
from PIL import Image
import torch
from torchvision import transforms as T
from torchvision.utils import save_image  # 저장을 위한 라이브러리 추가

# 설정 및 경로
INPUT_DIR = './data_sbP'
OUTPUT_DIR = './processed_data'
IMG_SIZE = 448 

# 폴더 초기화
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

# 전처리 파이프라인
def get_resnet_preprocess(size=224):
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def process_and_save():
    preprocess = get_resnet_preprocess(size=IMG_SIZE)
    
    # 파일 목록 가져오기
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"총 {len(files)}개의 파일을 처리합니다...")

    for i, file_name in enumerate(files):
        try:
            # 파일명 분석 및 폴더 생성
            parts = [p for p in file_name.split('_') if p]
            label = parts[0]
            label_dir = os.path.join(OUTPUT_DIR, label)
            os.makedirs(label_dir, exist_ok=True)

            # 이미지 로드 및 전처리
            img_path = os.path.join(INPUT_DIR, file_name)
            image = Image.open(img_path).convert('RGB')
            processed_data = preprocess(image)

            # [핵심] 실제 파일로 저장하는 로직
            # Normalize된 이미지는 수치가 변해있으므로 시각화용으로 저장할 때는 주의가 필요
            # 파일 생성 및 저장 확인을 위해 우선 저장이 되도록 작성
            output_path = os.path.join(label_dir, file_name)
            save_image(torch.as_tensor(processed_data), os.path.join(label_dir, file_name))
            if (i + 1) % 50 == 0:
                print(f"[{i + 1}/{len(files)}] 저장 완료...")
        
        except Exception as e:
            print(f"파일 처리 오류 ({file_name}): {e}")

    print(f"✅ 모든 데이터가 '{OUTPUT_DIR}' 폴더에 저장되었습니다!")

if __name__ == "__main__":
    process_and_save()